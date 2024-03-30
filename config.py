import argparse

def load_config():
    parser = argparse.ArgumentParser(description='Training Image-Restoration')
    parser.add_argument('--num_epoch', type=int,
                        default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size for training')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='path of checkpoint')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.0001, help='Learning rate')
    parser.add_argument('--save_checkpoint', type=str,
                        default=None, help='Path to save checkpoints')
    parser.add_argument('--num_frame', type=int, default=12,
                        help='number of frames for the model')
    args = parser.parse_args()

    return args