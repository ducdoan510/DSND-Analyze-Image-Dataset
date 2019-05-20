from torchvision import transforms, datasets
import torch
from PIL import Image
import numpy as np

def process_image(image):
    # resize the image
    width = image.width
    height = image.height
    size = (256, 256 * height // width) if height > width else (256 * width // height, 256)
    resized_image = image.resize(size)
    
    # crop center the image
    new_width = new_height = 224
    upper = (resized_image.height - new_height) / 2
    lower = (resized_image.height + new_height) / 2
    left = (resized_image.width - new_width) / 2
    right = (resized_image.width + new_width) / 2
    cropped_image = resized_image.crop((left, upper, right, lower))
    
    # convert to numpy array and normalize
    np_image = np.array(cropped_image) / 255
    mean_arr = np.array([0.485, 0.456, 0.406])
    std_arr = np.array([0.229, 0.224, 0.225])
    normalized_np_image = (np_image - mean_arr) / std_arr

    # transpose to pytorch format
    transposed_np_image = normalized_np_image.transpose(2, 0, 1)
    return transposed_np_image


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    image = Image.open(image_path)
    processed_image = process_image(image)
    logps = model.forward(torch.from_numpy(np.array([processed_image])).float().to(device))
    ps = torch.exp(logps)
    top_p, top_index = map(lambda t: t.tolist()[0], ps.topk(topk, dim=1))
    
    # convert index to class
    class_to_idx = model.class_to_idx
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    top_class = [idx_to_class[idx] for idx in top_index]
    
    return top_p, top_class


def load_images(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_validation_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    dataloaders = {
        'train': train_dataloaders,
        'valid': valid_dataloaders,
        'test': test_dataloaders
    }
    
    return dataloaders, train_dataset.class_to_idx