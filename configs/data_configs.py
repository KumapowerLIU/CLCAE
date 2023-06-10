from configs import transforms_config
from configs.paths_config import dataset_paths
import os

DATASETS = {
	'ffhq_encode_contrastive': {
		'transforms': transforms_config.ContrastiveTransforms,
		'train_image_root': os.path.join(dataset_paths['ffhq_generate_train'], 'image'),
		'train_latent_root': os.path.join(dataset_paths['ffhq_generate_train'], 'latent'),
		'test_image_root': os.path.join(dataset_paths['ffhq_generate_test'], 'image'),
		'test_latent_root': os.path.join(dataset_paths['ffhq_generate_test'], 'latent'),
		'avg_latent_root': dataset_paths['avg_latent_root'],
		'avg_image_root': dataset_paths['avg_image_root'],
	},

	'ffhq_encode_inversion': {
		'transforms': transforms_config.EncodeTransforms,
		'train_image_root': os.path.join(dataset_paths['ffhq_inversion'], 'train_img.txt'),
		'test_image_root': os.path.join(dataset_paths['ffhq_inversion'], 'val_img.txt'),
		'avg_image_root': dataset_paths['avg_image_root']
	},


	'car_encode_contrastive': {
		'transforms': transforms_config.ContrastiveTransforms,
		'train_image_root': os.path.join(dataset_paths['car_generate_train'], 'image'),
		'train_latent_root': os.path.join(dataset_paths['car_generate_train'], 'latent'),
		'test_image_root': os.path.join(dataset_paths['car_generate_test'], 'image'),
		'test_latent_root': os.path.join(dataset_paths['car_generate_test'], 'latent'),
		'avg_latent_root': dataset_paths['car_avg_latent_root'],
		'avg_image_root': dataset_paths['car_avg_image_root'],
	},

	'car_encode_inversion': {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_image_root': os.path.join(dataset_paths['car_inversion'], 'train_img.txt'),
		'test_image_root': os.path.join(dataset_paths['car_inversion'], 'val_img.txt'),
		'avg_image_root': dataset_paths['car_avg_image_root']
	},
}
