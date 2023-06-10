from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
from .image_folder import make_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagesDataset(Dataset):

    def __init__(self, image_root, image_avg_root, opts, image_transform=None):
        self.image_paths = sorted(make_dataset(dir=image_root, recursive=False, read_cache=True))
        self.image_transform = image_transform
        self.image_avg_root = image_avg_root
        self.opts = opts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if image_path.find('car_inversion') is not -1:
            image = image.crop((0, 64, 512, 448))
        image = self.image_transform(image)

        image_avg = Image.open(self.image_avg_root).convert('RGB')
        image_avg = self.image_transform(image_avg)
        return image, image_avg, image
