from torchvision import models
from torch import nn, optim
import os
import torch

NUMBER_OF_FLOWER_TYPES = 102

def get_customized_classifier(in_features, hidden_units):
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(hidden_units, NUMBER_OF_FLOWER_TYPES),
        nn.LogSoftmax(dim=1)
    )

def get_torchvision_model(model_name, hidden_units):
    model_class = getattr(models, model_name, None)
    if not model_class:
        return None
    model = model_class(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if not isinstance(model.classifier, nn.Sequential) or not isinstance(model.classifier[-1], nn.Linear):
        return None # only support model with multilayer classifier and the last layer is Linear
   
    model.classifier[-1] = get_customized_classifier(model.classifier[-1].in_features, hidden_units)
    return model
    
       
def train_model(learning_rate, model, epochs, device, dataloaders):
    train_dataloaders = dataloaders['train']
    valid_dataloaders = dataloaders['valid']
    criterion = nn.NLLLoss()
    
    if not model:
        raise ValueError('Model not supported')
    
    print('Learning rate:', learning_rate)
    
    # load and preprocess the pre-trained model
    model.to(device)
    
    # init necessary variable
    optimizer = optim.Adam(model.classifier[-1].parameters(), lr=learning_rate)
    running_loss = 0
    
    # training through number of epochs
    for epoch in range(epochs):
        for inputs, labels in train_dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward and backprop
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            # evaluate with validation dataset
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss
                    
                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print("Epoch: %d/%d" % (epoch + 1, epochs))
            print("Running loss: %.3f" % (running_loss / len(train_dataloaders)))
            avg_valid_loss, avg_accuracy = valid_loss / len(valid_dataloaders), accuracy / len(valid_dataloaders)
            print("Validation loss: %.3f, Validation accuracy: %.3f%%" % (avg_valid_loss, avg_accuracy * 100))
            
            running_loss = 0
            model.train()
    return avg_accuracy
    
    
def save_checkpoint(model, save_dir, hidden_units, model_name, class_to_idx):
    classifier_checkpoint = {
        'model_name': model_name,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
    }
    save_path = os.path.join(save_dir, "%s_%s.pth" % (model_name, hidden_units))
    torch.save(classifier_checkpoint, save_path)
    return save_path
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = get_torchvision_model(checkpoint['model_name'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model