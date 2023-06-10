import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from utils import data_utils
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
class ContrastiveDataset(Dataset):

	def __init__(self, image_root, latent_root, avg_latent_root, avg_image_root, opts, image_transform=None):
		self.image_paths = sorted(data_utils.make_dataset(image_root))
		self.target_paths = latent_root
		self.image_transform = image_transform
		self.opts = opts
		self.avg_latent_root = avg_latent_root
		self.avg_image_root = avg_image_root

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image_path = self.image_paths[index]
		image = Image.open(image_path).convert('RGB')
		name = os.path.split(image_path)[1].split('.')[0]
		latent_path = os.path.join(self.target_paths, name + '.npy')
		latent = torch.from_numpy(np.load(latent_path))
		image = self.image_transform(image)

		image_avg = Image.open(self.avg_image_root).convert('RGB')
		image_avg = self.image_transform(image_avg)
		latent_avg = torch.from_numpy(np.load(self.avg_latent_root))
		return image, latent.unsqueeze(0), image_avg, latent_avg.unsqueeze(0)
