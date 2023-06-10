from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # common option
        self.parser.add_argument('--train_inversion', action="store_true", help='Train your inversion model')
        self.parser.add_argument('--train_contrastive', action="store_true", help='Train your contrastive model')
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode_inversion', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=3, type=int,
                                 help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int,
                                 help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')

        self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        self.parser.add_argument('--n_styles', default=18)
        self.parser.add_argument('--num_latent', default=1)
        # for contrastive learning
        self.parser.add_argument('--contrastive_lambda', default=0.1, type=float, help='Contrastive loss factor')
        self.parser.add_argument('--use_norm', action='store_true', help='Use norm before calculate contrastive loss')
        self.parser.add_argument('--latent_embedding_dim', default=512, type=int, help='the dim of latent embedding')
        self.parser.add_argument('--image_embedding_dim', default=512, type=int, help='the dim of image_embedding')
        self.parser.add_argument('--projection_dim', default=512, type=int, help='projection dim of projection head')
        self.parser.add_argument('--load_pretrain_image_encoder', default=False)
        self.parser.add_argument('--checkpoint_path_image', default=None, type=str, help='Path to image model checkpoint')
        self.parser.add_argument('--checkpoint_path_latent', default=None, type=str, help='Path to latent model checkpoint')
        self.parser.add_argument('--checkpoint_path_af', default=None, type=str,
                                 help='Path to latset model checkpoint')

        self.parser.add_argument('--landmark_lambda', default=0.8, type=float, help='contrastive id loss multiplier factor' )
        self.parser.add_argument('--feature_matching_lambda', default=0.01, type=float,
                                 help='feature matching loss multiplier factor')
        self.parser.add_argument('--multi_scale_lpips', default=True)
        self.parser.add_argument('--not_for_feature', action='store_true')
        self.parser.add_argument('--not_for_ww', action='store_true')
        self.parser.add_argument('--global_step', default=None, type=int)
        self.parser.add_argument('--no_feature_attention', action='store_true')
        self.parser.add_argument('--contrastive_model_image', type=str, default='contrastive_ffhq_image')
        self.parser.add_argument('--contrastive_model_latent', type=str, default='contrastive_ffhq_latent')
        self.parser.add_argument('--no_w_attention', action='store_true')
        self.parser.add_argument('--idx_k', type=int, default=None)
        self.parser.add_argument('--no_res', action='store_true')
        ####################################
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')
        self.parser.add_argument('--moco_lambda', default=0, type=float,
                                 help='Moco-based feature similarity loss multiplier factor')

        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_epoch', default=80, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

        # ddp
        self.parser.add_argument("--local_rank", default=0, type=int)
        self.parser.add_argument("--use_ddp", action='store_true', help = 'Whether to use the ddp of pytorch')

        # arguments for weights & biases support
        self.parser.add_argument('--use_wandb', action="store_true",
                                 help='Whether to use Weights & Biases to track experiment.')


    def parse(self):
        opts = self.parser.parse_args()
        return opts
