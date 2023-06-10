import os
import argparse
import numpy as np
from fid import FID
from PIL import Image
from natsort import natsorted
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
# import lpips
import shutil


def img_to_tensor(img):
    image_size = img.width
    image = (np.asarray(img) / 255.0).reshape(image_size * image_size, 3).transpose().reshape(3, image_size, image_size)
    torch_image = torch.from_numpy(image).float()
    torch_image = torch_image * 2.0 - 1.0
    torch_image = torch_image.unsqueeze(0)
    return torch_image


class Reconstruction_Metrics:
    def __init__(self, metric_list=['ssim', 'psnr', 'fid'], data_range=1, win_size=21, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        self.fid_calculate = FID()
        # self.loss_fn_vgg = lpips.LPIPS(net='alex')
        for metric in metric_list:
            setattr(self, metric, True)

    def calculate_metric(self, real_image_path, fake_image_path):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """
        # with torch.no_grad():
        #     fid_value = self.fid_calculate.calculate_from_disk(fake_image_path, real_image_path)
        psnr = []
        ssim = []
        # lipis = []
        image_name_list = [name for name in os.listdir(real_image_path) if
                           name.endswith((('.png', '.jpg', '.jpeg', '.JPG', '.bmp')))]
        # fake_image_name_list = os.listdir(fake_image_path)
        for i, image_name in enumerate(image_name_list):
            image_fake_name = image_name.split('.')[0] + '.jpg'
            path_real = os.path.join(real_image_path, image_name)
            path_fake = os.path.join(fake_image_path, image_fake_name)
            PIL_real = Image.open(path_real).convert('RGB')
            PIL_fake = Image.open(path_fake).convert('RGB')
            # PIL_real = PIL_real.resize((256, 192))
            # PIL_fake =  PIL_fake.resize((256, 256))
            # fake_torch_image = img_to_tensor(PIL_fake)
            # real_torch_image = img_to_tensor(PIL_real)
            img_content_real = np.array(PIL_real).astype(np.float32) / 255.0
            img_content_fake = np.array(PIL_fake).astype(np.float32) / 255.0
            # img_content_fake = img_content_fake[32:224, :, :]
            # print(img_content_fake.shape)
            psnr_each_img = peak_signal_noise_ratio(img_content_real, img_content_fake)
            ssim_each_image = structural_similarity(img_content_real, img_content_fake, data_range=self.data_range,
                                                    win_size=self.win_size, multichannel=self.multichannel)
            # lipis_each_image = self.loss_fn_vgg(fake_torch_image, real_torch_image)
            # lipis_each_image = lipis_each_image.detach().numpy()
            psnr.append(psnr_each_img)
            ssim.append(ssim_each_image)
            # lipis.append(lipis_each_image)
        print(
            "PSNR: %.4f" % np.round(np.mean(psnr), 4),
            "PSNR Variance: %.4f" % np.round(np.var(psnr), 4))
        print(
            "SSIM: %.4f" % np.round(np.mean(ssim), 4),
            "SSIM Variance: %.4f" % np.round(np.var(ssim), 4))
        # print(
        #     "LPIPS: %.4f" % np.round(np.mean(lipis), 4),
        #     "LPIPS Variance: %.4f" % np.round(np.var(lipis), 4))
        return np.round(np.mean(psnr), 4), np.round(np.mean(ssim), 4)




def select_fake_from_fewshot(video_name, start, nums):
    base_path = r'D:\My Documents\Desktop\transformer_dance\results\LWG\result\reconstruct'
    video_path = os.path.join(base_path, video_name, 'imitators')
    file_list = natsorted(os.listdir(video_path))
    select_file_list = file_list[start:start + nums]
    save_base = r'D:\My Documents\Desktop\transformer_dance\results\real'
    save_file = os.path.join(save_base, video_name, 'fake')
    if os.path.exists(save_file) is False:
        os.makedirs(save_file)
    for i, file_name in enumerate(select_file_list):
        original_path = os.path.join(video_path, file_name)
        save_path = os.path.join(save_file, f'{str(i + 1).zfill(6)}.png')
        shutil.copyfile(original_path, save_path)



def get_metric(fake_dir, real_dir):
    Get_metric = Reconstruction_Metrics()
    psnr_out, ssim_out = Get_metric.calculate_metric(fake_dir, real_dir)
    save_txt = os.path.join(fake_dir, 'metric.txt')
    with open(save_txt, 'a') as txt2:
        txt2.write("psnr:")
        txt2.write(str(psnr_out))
        txt2.write("  ")
        txt2.write("ssim:")
        txt2.write(str(ssim_out))
        txt2.write('\n')


# def get_metric(fake_dir, real_dir):
#     Get_metric = Reconstruction_Metrics()
#     psnr_out, ssim_out, lipis_out, fid_value_out = Get_metric.calculate_metric(fake_dir, real_dir)
#     save_txt = os.path.join(fake_dir, 'metric.txt')
#     with open(save_txt, 'a') as txt2:
#         txt2.write("psnr:")
#         txt2.write(str(psnr_out))
#         txt2.write("  ")
#         txt2.write("ssim:")
#         txt2.write(str(ssim_out))
#         txt2.write(" ")
#         txt2.write("lipis:")
#         txt2.write(str(lipis_out))
#         txt2.write(" ")
#         txt2.write("fid:")
#         txt2.write(str(fid_value_out))
#         txt2.write('\n')


if __name__ == "__main__":
    real_path = r'/apdcephfs/share_1290939/kumamzqliu/data/face_inversion/test'
    fake_path = r'/apdcephfs/share_1290939/kumamzqliu/resutls/ours/inference_w'
    get_metric(real_path, fake_path)