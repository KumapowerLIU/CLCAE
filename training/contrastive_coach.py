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
from criteria import contrastive_loss
from configs import data_configs
from datasets.contrastive_dataset import ContrastiveDataset
from criteria.lpips.lpips import LPIPS
from models.image_encoder import ImageEncoder
from models.latent_encoder import LatentEncoder
from training.ranger import Ranger
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.test_ddp_sample import SequentialDistributedSampler
import torch.distributed as dist


class ContrastiveCoach:
    def __init__(self, opts):
        self.opts = opts
        self.epoch = 0
        self.global_step = 0
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.opts.device = self.device

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        # Initialize network

        self.net_image = ImageEncoder(self.opts)
        self.net_latent = LatentEncoder(self.opts)
        self.net_image.init_weights()
        self.net_latent.init_weights()
        self.net_image.load_weights()
        self.net_latent.load_weights()
        self.net_latent.to(self.device)
        self.net_image.to(self.device)
        # Initialize loss
        if self.opts.use_ddp:
            torch.cuda.empty_cache()
            # dist.init_process_group(backend="nccl")
            print('###########################DDP initialized ########################################')
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])

            print("The nums of current GPUs:", torch.cuda.device_count())
            dist.init_process_group(backend="nccl", init_method='env://', rank=self.rank, world_size=self.world_size)
            print('###########################DDP initialized ########################################')
            print('world_sized:', self.world_size)
            self.net_image = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_image)
            self.net_image = DDP(self.net_image, device_ids=[self.local_rank], output_device=self.local_rank,
                                 find_unused_parameters=True)
            self.net_latent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_latent)
            self.net_latent = DDP(self.net_latent, device_ids=[self.local_rank],
                                  find_unused_parameters=True)
            self.contrastive_loss = contrastive_loss.ClipLoss(cache_labels=True, rank=self.local_rank,
                                                              world_size=self.world_size)
        else:
            self.contrastive_loss = contrastive_loss.ClipLoss().to(self.device)

        # Initialize optimizer
        self.optimizer_image, self.optimizer_latent = self.configure_optimizers()
        self.lr_scheduler_image = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_image, mode="min", patience=10, factor=0.1
        )
        self.lr_scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_latent, mode="min", patience=10, factor=0.1
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

    def train(self):
        self.net_image.train()
        self.net_latent.train()
        while self.global_step < self.opts.max_steps:
            loss_meter = train_utils.AvgMeter()
            if self.opts.use_ddp:
                self.train_dataloader.sampler.set_epoch(self.epoch)
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer_image.zero_grad()
                self.optimizer_latent.zero_grad()
                # x is image, y is latent
                x, y, x_avg, y_avg = batch
                x, y, x_avg, y_avg = x.to(self.device).float(), y.to(self.device).float(), x_avg.to(
                    self.device).float(), y_avg.to(self.device).float()
                x_embedding, t = self.net_image.forward(x, x_avg)
                y_embedding = self.net_latent.forward(y, y_avg)

                loss, loss_dict, id_logs = self.calc_loss(x_embedding, y_embedding, t)
                loss.backward()
                self.optimizer_image.step()
                self.optimizer_latent.step()

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
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
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

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
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
            self.lr_scheduler_image.step(loss_meter.get())
            self.lr_scheduler_latent.step(loss_meter.get())

    def validate(self):
        self.net_image.eval()
        self.net_latent.eval()
        agg_loss_dict = []
        agg_loss_value = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                x, y, x_avg, y_avg = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                x_avg, y_avg = x_avg.to(self.device).float(), y_avg.to(self.device).float()
                x_embedding, t = self.net_image.forward(x, x_avg)
                y_embedding = self.net_latent.forward(y, y_avg)
                loss, loss_dict, id_logs = self.calc_loss(x_embedding, y_embedding, t)
                agg_loss_dict.append(loss_dict)
                agg_loss_value.append(torch.tensor(loss).unsqueeze(0))
                # For first step just do sanity test on small amount of data
                if self.global_step == 0 and batch_idx >= 4:
                    self.net_image.train()
                    self.net_latent.train()
                    return None  # Do not log, inaccurate in first batch
            if self.opts.use_ddp:
                loss_avg_all = train_utils.distributed_concat(torch.concat(agg_loss_value, dim=0),
                                                              len(self.sampler_test.dataset))
                loss_avg = sum(loss_avg_all) / len(loss_avg_all)
                loss_dict['loss'] = float(loss_avg)
                loss_dict['loss_contrastive'] = float(loss_avg)

            else:
                loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
            self.log_metrics(loss_dict, prefix='test')
            self.print_metrics(loss_dict, prefix='test')
            self.net_image.train()
            self.net_latent.train()
            return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_image_name = 'best_model_image.pt' if is_best else f'iteration_image_{self.global_step}.pt'
        save_latent_name = 'best_model_latent.pt' if is_best else f'iteration_latent_{self.global_step}.pt'
        save_dict_image = self.__get_save_dict(self.net_image.module)
        save_dict_latent = self.__get_save_dict(self.net_latent.module)
        checkpoint_path_image = os.path.join(self.checkpoint_dir, save_image_name)
        checkpoint_path_latent = os.path.join(self.checkpoint_dir, save_latent_name)
        torch.save(save_dict_image, checkpoint_path_image)
        torch.save(save_dict_latent, checkpoint_path_latent)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        params_image = list(self.net_image.parameters())
        params_latent = list(self.net_latent.parameters())
        if self.opts.optim_name == 'adam':
            optimizer_image = torch.optim.Adam(params_image, lr=self.opts.learning_rate)
            optimizer_latent = torch.optim.Adam(params_latent, lr=self.opts.learning_rate)
        else:
            optimizer_image = Ranger(params_image, lr=self.opts.learning_rate)
            optimizer_latent = Ranger(params_latent, lr=self.opts.learning_rate)

        return optimizer_image, optimizer_latent

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ContrastiveDataset(image_root=dataset_args['train_image_root'],
                                           latent_root=dataset_args['train_latent_root'],
                                           avg_image_root=dataset_args['avg_image_root'],
                                           avg_latent_root=dataset_args['avg_latent_root'],
                                           image_transform=transforms_dict['transform_gt_train'],
                                           opts=self.opts)
        test_dataset = ContrastiveDataset(image_root=dataset_args['test_image_root'],
                                          latent_root=dataset_args['test_latent_root'],
                                          avg_image_root=dataset_args['avg_image_root'],
                                          avg_latent_root=dataset_args['avg_latent_root'],
                                          image_transform=transforms_dict['transform_test'],
                                          opts=self.opts)
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, t):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        loss_contrastive = self.contrastive_loss(x, y, t)
        loss_dict['loss_contrastive'] = float(loss_contrastive)
        loss += loss_contrastive
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)


    def __get_save_dict(self, net):
        save_dict = {
            'state_dict': net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training

        return save_dict
