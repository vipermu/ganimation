import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True  # Improves runtime if the input size is constant

    # Set the outputs path
    if config.mode == 'test':
        config.outputs_dir = os.path.join('tests', config.outputs_dir)
    config.log_dir = os.path.join(config.outputs_dir, config.log_dir)
    config.model_save_dir = os.path.join(config.outputs_dir, config.model_save_dir)
    config.sample_dir = os.path.join(config.outputs_dir, config.sample_dir)
    config.result_dir = os.path.join(config.outputs_dir, config.result_dir)

    initialize_directories(config)

    dataset_loader = get_loader(config.image_dir, config.attr_path, config.c_dim,
                                config.image_size, config.batch_size, config.mode,
                                config.num_workers)

    solver = Solver(dataset_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


def initialize_directories(config):
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)


def str2bool(v):
    return v.lower() in ('true')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=17,
                        help='dimension of domain labels')

    parser.add_argument('--image_size', type=int,
                        default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64,
                        help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64,
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6,
                        help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=160,
                        help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10,
                        help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='weight for gradient penalty')
    parser.add_argument('--lambda_sat', type=float, default=0.1,
                        help='weight for attention saturation loss')
    parser.add_argument('--lambda_smooth', type=float, default=1e-4,
                        help='weight for the attention smoothing loss')

    # Training configuration.
    parser.add_argument('--batch_size', type=int,
                        default=1, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of total epochs for training D')
    parser.add_argument('--num_epochs_decay', type=int, default=20,
                        help='number of epochs for start decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='number of D updates per each G update')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int,
                        default=None, help='resume training from this step')
    parser.add_argument('--first_epoch', type=int,
                        default=0, help='First epoch')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--use_virtual', type=str2bool, default=False,
                        help='Boolean to decide if we should use the virtual cycle concistency loss')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--num_sample_targets', type=int, default=4,
                        help="number of targets to use in the samples visualization")

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str,
                        default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--outputs_dir', type=str, default='experiments')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
