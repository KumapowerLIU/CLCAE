import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import math

sys.path.append(".")
sys.path.append("..")
from editings import latent_editor
from configs import data_configs
from datasets.inference_dataset_me import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.attention_feature_psp import AFPSP
from configs.paths_config import edit_paths

# from editings
Latent_edit_f = latent_editor.LatentEditor()
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

def run():
    test_opts = TestOptions().parse()
    edit_directory_path = os.path.join(test_opts.exp_dir, test_opts.edit_attribute)
    os.makedirs(edit_directory_path, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path_af, map_location='cpu')
    opts = ckpt['opts']
    print(f"iter:{ckpt['iter']}")
    iter = 1.0
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    print(opts)
    print('#################### network init #####################')
    net = AFPSP(opts)
    net.load_weights()
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               image_avg_root=dataset_args['avg_image_root'],
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)
    print(f'######edit {opts.edit_attribute} ##############')
    edit_degree = opts.edit_degree
    edit_direction, ganspace_pca = edit(opts)

    global_i = 0
    global_time = []

    os.makedirs(edit_directory_path, exist_ok=True)
    out_path_results = os.path.join(edit_directory_path, 'inference_results')
    out_path_coupled = os.path.join(edit_directory_path, 'inference_coupled')
    out_path_w = os.path.join(edit_directory_path, 'inference_w')
    out_path_ww = os.path.join(edit_directory_path, 'inference_ww')
    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(out_path_w, exist_ok=True)
    os.makedirs(out_path_ww, exist_ok=True)
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda, image_avg = input_batch
            input_cuda = input_cuda.cuda().float()
            image_avg = image_avg.cuda().float()
            tic = time.time()
            result_batch, results_ww_batch, results_w_batch, latent_base = run_on_batch(input_cuda, image_avg, net,
                                                                                        opts, iter, ganspace_pca, edit_direction)
            toc = time.time()
            global_time.append(toc - tic)
        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            result_ww = tensor2im(results_ww_batch[i])
            result_w = tensor2im(results_w_batch[i])

            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_cuda[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                # otherwise, save the original and output
                res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                      np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            im_save_path_ww = os.path.join(out_path_ww, os.path.basename(im_path))
            im_save_path_w = os.path.join(out_path_w, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)
            Image.fromarray(np.array(result_ww)).save(im_save_path_ww)

            Image.fromarray(np.array(result_w)).save(im_save_path_w)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def edit(opts):
    ganspace_pca = None
    if opts.edit_attribute == 'age' or opts.edit_attribute == 'smile' or opts.edit_attribute == 'pose':
        edit_direction = torch.load(edit_paths[opts.edit_attribute]).cuda()

    elif opts.edit_attribute == 'eyes' or opts.edit_attribute == 'beard' or opts.edit_attribute == 'lip' :
        ganspace_pca = torch.load(edit_paths['ffhq_pca'])
        ganspace_directions = {
            'eyes': (54, 7, 8, 20),
            'beard': (58, 7, 9, 20),
            'lip': (34, 10, 11, 20)}
        edit_direction = ganspace_directions[opts.edit_attribute]
    else:
        ganspace_pca = torch.load(edit_paths['car_pca'])
        ganspace_directions = {
            'Viewpoint1': (0, 0, 5, 2),
            'Viewpoint2': (0, 0, 5, -2),
            'Cube': (16, 3, 6, 25),
            'Color': (22, 9, 11, -27),
            'Grass': (41, 9, 11, -18)}
        edit_direction = ganspace_directions[opts.edit_attribute]
    # For a single edit:
    return edit_direction, ganspace_pca


def run_on_batch(inputs, image_avg, net, opts, iter, ganspace_pca, edit_direction):
    log_size = int(math.log(opts.output_size, 2))
    num_layers = (log_size - 2) * 2 + 1

    result_batch, latent_refine, latent_base, feature_offset, feature_refine, results_ww, results_w = net(inputs,
                                                                                                          image_avg,
                                                                                                          randomize_noise=False,
                                                                                                          resize=opts.resize_outputs,
                                                                                                          interation=iter,
                                                                                                          return_features=True)

    latent_edit = Latent_edit_f.apply_ganspace(latent_refine, ganspace_pca, [edit_direction])
    _, features_edit = net.decoder([latent_edit],
                                   input_is_latent=True,
                                   return_features=True,
                                   randomize_noise=False,
                                   return_latents=False)

    features_offsets = [None] * 5 + [
        feature_offset + features_edit[5] - feature_refine[0]] + [None] * (num_layers - 5)

    result_batch, features_refine = net.decoder([latent_edit],
                                                input_is_latent=True,
                                                return_features=True,
                                                randomize_noise=False,
                                                return_latents=False,
                                                features_in=features_offsets,
                                                feature_scale=1.0)

    results_ww, _ = net.decoder([latent_edit],
                                input_is_latent=True,
                                return_features=True,
                                randomize_noise=False,
                                return_latents=False)

    results_w, _ = net.decoder([latent_edit],
                               input_is_latent=True,
                               return_features=True,
                               randomize_noise=False,
                               return_latents=False)

    if opts.resize_outputs:

        results_ww = face_pool(results_ww)
        results_w = face_pool(results_w)
        result_batch = face_pool(result_batch)

    if opts.is_car:
        results_ww = results_ww[:, :, 32:224, :]
        results_w = results_w[:, :, 32:224, :]
        result_batch = result_batch[:, :, 32:224, :]

    return result_batch, results_ww, results_w, latent_base


if __name__ == '__main__':
    run()
