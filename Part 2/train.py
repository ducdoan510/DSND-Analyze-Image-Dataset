import argparse
import torch
from model import get_torchvision_model, train_model, save_checkpoint
from image import load_images
from workspace_utils import active_session


parser = argparse.ArgumentParser(description='Training image classifier')
parser.add_argument('data_directory', help='image data directory path with train/valid/test subfolders')
parser.add_argument('--save_dir', help='Model checkpoint saving', default='.')
parser.add_argument('--arch', help='Pretrained model from torchvision', default='vgg16')
parser.add_argument('--learning_rate', type=float, help='Optimizer learning rate', default=0.003)
parser.add_argument('--hidden_units', type=int, help='Number of hidden units in customized classifier', default=256)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=30)
parser.add_argument('--gpu', action='store_true', default=False, help='Flag to set using GPU')

args = parser.parse_args()
print('Hyperparameters:', args)

dataloaders, class_to_idx = load_images(args.data_directory)


device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
with active_session():
    print("Start loading model...")
    model = get_torchvision_model(args.arch, args.hidden_units)
    print(model.classifier)
    print("Start training model...")
    train_model(args.learning_rate, model, args.epochs, device, dataloaders)

    
save_path = save_checkpoint(model, args.save_dir, args.hidden_units, args.arch, class_to_idx)
print('Training complete. Model checkpoint is save at: %s' % save_path)
