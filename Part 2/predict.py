import argparse
import json
from model import load_checkpoint
from image import predict
import torch

parser = argparse.ArgumentParser(description='Predict image type')
parser.add_argument('image_path', help='path to image to predict')
parser.add_argument('checkpoint', help='path to model checkpoint')
parser.add_argument('--top_k', help='Number of top classes to display', default=1, type=int)
parser.add_argument('--category_name', help='path to json file representing mapping flower category to actual name')
parser.add_argument('--gpu', action='store_true', default=False, help='Flag to set using GPU')

args = parser.parse_args()
print('Hyperparameters:', args)

model = load_checkpoint(args.checkpoint)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

top_p, top_class = predict(args.image_path, model, device, args.top_k)
print("Top %d predict result" % args.top_k)
if args.category_name:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    top_name = [cat_to_name[cat] for cat in top_class]
    for p, name in zip(top_p, top_name):
        print("%20s\t%.4f" % (name, p)) 
else:
    for p, cl in zip(top_p, top_class):
        print("%3s\t%.4f" % (cl, p))
    
    
