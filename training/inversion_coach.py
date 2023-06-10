import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random
from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss, contrastive_id_loss, feature_matching_loss
from configs import data_configs
from datasets.inversion_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
# from models.image_encoder import ImageEncoder
# from models.latent_encoder import LatentEncoder
from models.attention_feature_psp import AFPSP
from training.ranger import Ranger
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.test_ddp_sample import SequentialDistributedSampler
import torch.distributed as dist


def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class InversionCoach:
    def __init__(self, opts):
        self.opts = opts
        self.epoch = 0
        if self.opts.global_step is not None:
            self.global_step = self.opts.global_step
        else:
            self.global_step = 0
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.opts.device = self.device
        self.scale_mode = 'bilinear'
        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)
        self.iter = None
        # Initialize network

        self.net = AFPSP(self.opts)
        self.net.init_weights()
        self.net.load_weights()
        self.net.to(self.device)
        if self.net.latent_avg is None:
            raise ValueError('check your style gan para. or your stylegan not the face')

        print('################### Net init done ####################################')
        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
            print('################### lpips_loss init done ####################################')
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
            print('################### id_loss init done ####################################')
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
            print('################### w_norm_loss init done ####################################')
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()
            print('################### moco_lambda  init done ####################################')
        if self.opts.contrastive_lambda > 0:
            self.contrastive_id_loss = contrastive_id_loss.ContrastiveID(opts).to(self.device)
            print('################### contrastive_id_loss  init done ####################################')
        print('################### Loss init done ####################################')

        if self.opts.use_ddp:
            torch.cuda.empty_cache()
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])

            print("The nums of current GPUs:", torch.cuda.device_count())
            dist.init_process_group(backend="nccl", init_method='env://', rank=self.rank, world_size=self.world_size)
            print('###########################DDP initialized ########################################')
            print('world_sized:', self.world_size)
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = DDP(self.net, device_ids=[self.local_rank], output_device=self.local_rank,
                           find_unused_parameters=True)
            for name, parameter in self.net.module.encoder.named_parameters():
                print(name, parameter.requires_grad)

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, factor=0.1
        )

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        if self.opts.use_ddp:
            sampler_train = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            self.sampler_test = SequentialDistributedSampler(self.test_dataset, batch_size=self.opts.batch_size)
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=self.opts.batch_size,
                                               shuffle=False,
                                               num_workers=int(self.opts.workers),
                                               drop_last=True,
                                               sampler=sampler_train)
            self.test_dataloader = DataLoader(self.test_dataset,
                                              batch_size=self.opts.test_batch_size,
                                              shuffle=False,
                                              num_workers=int(self.opts.test_workers),
                                              drop_last=True,
                                              sampler=self.sampler_test)
        else:
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=self.opts.batch_size,
                                               shuffle=True,
                                               num_workers=int(self.opts.workers),
                                               drop_last=True)
            self.test_dataloader = DataLoader(self.test_dataset,
                                              batch_size=self.opts.test_batch_size,
                                              shuffle=False,
                                              num_workers=int(self.opts.test_workers),
                                              drop_last=True)
        print('################### Dataset init done ####################################')
        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps
        print('################### Checkpoint init done ####################################')

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            loss_meter = train_utils.AvgMeter()
            if self.opts.use_ddp:
                self.train_dataloader.sampler.set_epoch(self.epoch)
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                # x is image, x=y
                x, x_avg, y = batch
                x, x_avg, y = x.to(self.device).float(), x_avg.to(
                    self.device).float(), y.to(self.device).float()
                # self.iter = 1
                # self.iter = max(batch_idx / (len(self.train_dataloader) - 1), 0.55)
                self.iter = batch_idx / (len(self.train_dataloader) - 1)
                x_rec, latent_refine, latent_base, feature_offset, feature_refine, image_ww, image_w = self.net.forward(
                    x, x_avg,
                    return_features=True,
                    interation=self.iter)

                loss, loss_dict, id_logs = self.calc_loss(x, y, x_rec, x_avg, latent_refine, latent_base,
                                                          feature_offset, feature_refine, image_ww, image_w)
                loss.backward()
                self.optimizer.step()
                if self.opts.use_ddp:
                    if self.local_rank == 0:
                        if self.global_step % self.opts.image_interval == 0 or (
                                self.global_step < 1000 and self.global_step % 25 == 0):
                            self.parse_and_log_images(id_logs, x, y, x_rec, image_ww, image_w, title='images/train/faces')
                else:
                    if self.global_step % self.opts.image_interval == 0 or (
                            self.global_step < 1000 and self.global_step % 25 == 0):
                        self.parse_and_log_images(id_logs, x, y, x_rec, image_ww, image_w, title='images/train/faces')
                # Logging related
                if self.global_step % self.opts.board_interval == 0:
                    if self.opts.use_ddp:
                        if self.local_rank == 0:
                            self.print_metrics(loss_dict, prefix='train')
                            self.log_metrics(loss_dict, prefix='train')
                    else:
                        self.print_metrics(loss_dict, prefix='train')
                        self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps or self.iter >= 1 or self.iter == 0:
                    val_loss_dict = self.validate()
                    if val_loss_dict is not None:
                        loss_meter.update(val_loss_dict['loss'], len(self.test_dataset))
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        if self.opts.use_ddp:
                            if self.local_rank == 0:
                                self.checkpoint_me(val_loss_dict, is_best=True)
                        else:
                            self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps or self.iter >= 1:
                    if val_loss_dict is not None:
                        if self.opts.use_ddp:
                            if self.local_rank == 0:
                                self.checkpoint_me(val_loss_dict, is_best=False)
                        else:
                            self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        if self.opts.use_ddp:
                            if self.local_rank == 0:
                                self.checkpoint_me(loss_dict, is_best=False)
                        else:
                            self.checkpoint_me(loss_dict, is_best=False)
                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
            self.epoch = self.epoch + 1
            self.lr_scheduler.step(loss_meter.get())

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        # agg_loss_value = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                x, x_avg, y = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                x_avg = x_avg.to(self.device).float()
                if self.iter is None:
                    iter = 2
                else:
                    iter = self.iter
                x_rec, latent_refine, latent_base, feature_offset, feature_refine, image_ww, image_w = self.net.forward(
                    x, x_avg,
                    return_features=True,
                    interation=iter)

                loss, loss_dict, id_logs = self.calc_loss(x, y, x_rec, x_avg, latent_refine, latent_base,
                                                          feature_offset, feature_refine, image_ww, image_w)
                if self.opts.use_ddp:
                    if self.local_rank == 0:
                        self.parse_and_log_images(id_logs, x, y, x_rec, image_ww, image_w,
                                                  title='images/test/faces',
                                                  subscript='{:04d}'.format(batch_idx))
                else:
                    self.parse_and_log_images(id_logs, x, y, x_rec, image_ww, image_w,
                                              title='images/test/faces',
                                              subscript='{:04d}'.format(batch_idx))
                agg_loss_dict.append(loss_dict)
                # agg_loss_value.append(torch.tensor(loss).unsqueeze(0))
                # For first step just do sanity test on small amount of data
                if self.global_step == 0 and batch_idx >= 4:
                    self.net.train()
                    return None  # Do not log, inaccurate in first batch
            if self.opts.use_ddp:
                for loss_name in loss_dict.keys():
                    agg_loss_value = []
                    for output in agg_loss_dict:
                        agg_loss_value.append(torch.tensor(output[loss_name]).unsqueeze(0).to(self.device))

                    loss_avg_all = train_utils.distributed_concat(torch.concat(agg_loss_value, dim=0),
                                                                  len(self.sampler_test.dataset))
                    loss_avg = sum(loss_avg_all) / len(loss_avg_all)
                    loss_dict[loss_name] = float(loss_avg)

            else:
                loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
            if self.opts.use_ddp:
                if self.local_rank == 0:
                    self.log_metrics(loss_dict, prefix='test')
                    self.print_metrics(loss_dict, prefix='test')
            else:
                self.log_metrics(loss_dict, prefix='test')
                self.print_metrics(loss_dict, prefix='test')
            self.net.train()
            return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}_iter_{self.iter}.pt'
        save_dict_image = self.__get_save_dict(self.net.module)
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict_image, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    f'**Best**: Step - {self.global_step}, Iter - {self.iter}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, Iter - {self.iter}, \n{loss_dict}\n')

    def configure_optimizers(self):
        params = list(self.net.module.encoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(filter(lambda p: p.requires_grad, self.net.module.encoder.parameters()),
                               lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(image_root=dataset_args['train_image_root'],
                                      image_avg_root=dataset_args['avg_image_root'],
                                      image_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(image_root=dataset_args['test_image_root'],
                                     image_avg_root=dataset_args['avg_image_root'],
                                     image_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, x_rec, x_avg, latent_refine, latent_base, feature_offset, feature_refine,
                  image_ww, image_w):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(x_rec, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
            if image_ww is not None:
                loss_id_ww, sim_improvement_ww, id_logs_ww = self.id_loss(image_ww, y, x)
                loss_dict['loss_id_ww'] = float(loss_id_ww)
                loss_dict['id_improve_ww'] = float(sim_improvement_ww)
                loss += loss_id_ww * self.opts.id_lambda
            if image_w is not None:
                loss_id_w, sim_improvement_w, id_logs_w = self.id_loss(image_w, y, x)
                loss_dict['loss_id_w'] = float(loss_id_w)
                loss_dict['id_improve_w'] = float(sim_improvement_w)
                loss += loss_id_w * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(x_rec, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
            if image_ww is not None:
                loss_l2_ww = F.mse_loss(image_ww, y)
                loss_dict['loss_l2_ww'] = float(loss_l2_ww)
                loss += loss_l2_ww * self.opts.l2_lambda
            if image_w is not None:
                loss_l2_w = F.mse_loss(image_w, y)
                loss_dict['loss_l2_w'] = float(loss_l2_w)
                loss += loss_l2_w * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            if self.opts.multi_scale_lpips:
                for k in range(3):
                    loss_lpips += self.lpips_loss(downscale(x_rec, k, self.scale_mode),
                                                  downscale(y, k, self.scale_mode))
                else:
                    loss_lpips = self.lpips_loss(x_rec, y)

            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
            if image_ww is not None:
                loss_lpips_ww = 0
                if self.opts.multi_scale_lpips:
                    for k in range(3):
                        loss_lpips_ww += self.lpips_loss(downscale(image_ww, k, self.scale_mode),
                                                           downscale(y, k, self.scale_mode))
                    else:
                        loss_lpips_ww = self.lpips_loss(image_ww, y)
                loss_dict['loss_lpips_ww'] = float(loss_lpips_ww)
                loss += loss_lpips_ww * self.opts.lpips_lambda
            if image_w is not None:
                loss_lpips_w = 0
                if self.opts.multi_scale_lpips:
                    for k in range(3):
                        loss_lpips_w += self.lpips_loss(downscale(image_w, k, self.scale_mode),
                                                           downscale(y, k, self.scale_mode))
                    else:
                        loss_lpips_w = self.lpips_loss(image_w, y)
                loss_dict['loss_lpips_w'] = float(loss_lpips_w)
                loss += loss_lpips_w * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(x[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(x[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent_refine, self.net.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(x_rec, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda
            if image_ww is not None:
                loss_moco_ww, sim_improvement_ww, id_logs_ww = self.moco_loss(image_ww, y, x)
                loss_dict['loss_moco_ww'] = float(loss_moco_ww)
                loss_dict['id_improve_ww'] = float(sim_improvement_ww)
                loss += loss_moco_ww * self.opts.moco_lambda
            if image_w is not None:
                loss_moco_w, sim_improvement_w, id_logs_w = self.moco_loss(image_w, y, x)
                loss_dict['loss_moco_w'] = float(loss_moco_w)
                loss_dict['id_improve_w'] = float(sim_improvement_w)
                loss += loss_moco_w * self.opts.moco_lambda
        if self.opts.feature_matching_lambda > 0:
            loss_feature = self.mse_loss(feature_offset, feature_refine)
            loss_dict['loss_feature'] = float(loss_feature)
            loss += loss_feature * self.opts.feature_matching_lambda

        if self.opts.contrastive_lambda > 0 and image_w is not None:
            if len(latent_base.shape) == 3:
                latent_base = latent_base[:, 0:1, :]
            elif len(latent_base.shape) == 3:
                latent_base = latent_base.unsqueeze(1)
            else:
                raise ValueError('please check the shape of base latent')
            loss_contras = self.contrastive_id_loss(image_w, x_avg, latent_base, self.net.module.latent_avg)
            loss_dict['loss_contras'] = float(loss_contras)
            loss += loss_contras * self.opts.contrastive_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}, iter {self.iter}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, image_ww, image_w, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if image_w is not None:
                cur_im_data['output_face_w'] = common.tensor2im(image_w[i])
            else:
                cur_im_data['output_face_w'] = common.tensor2im(y_hat[i])
            if image_ww is not None:
                cur_im_data['output_face_ww'] = common.tensor2im(image_ww[i])
            else:
                cur_im_data['output_face_ww'] = common.tensor2im(y_hat[i])
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        iter = self.iter
        if log_latest:
            step = 0
            iter = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}_{iter:.4f}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}_{iter:.4f}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self, net):
        save_dict = {
            'state_dict': net.state_dict(),
            'opts': vars(self.opts),
            'iter': self.iter
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training

        return save_dict
