from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from .image_folder import make_dataset

class InferenceDataset(Dataset):

	def __init__(self, root, opts, image_avg_root, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts
		self.image_avg_root = image_avg_root

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		image_avg = Image.open(self.image_avg_root).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
			image_avg = self.transform(image_avg)
		return from_im, image_avg
