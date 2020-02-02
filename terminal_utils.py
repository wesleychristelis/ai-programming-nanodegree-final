import argparse
from PIL import Image
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
import json
from collections import OrderedDict
from time import time, sleep, localtime, strftime
import os

def test_helper():
    return "Helper is imported"

def get_command_args_train():
    """
    Get argument parsed for training
    
    ArgumentParser object. 
       data_dir - Path to the image files
       arch - pretrained CNN model architecture to use for image classification (default-
              pick any of the following vgg, densenet)
       save_dir - Set directory to save checkpoints
       learning_rate - learning rate for optimizer
       hidden_units - number of hidden units
       epochs - number of epochs
       gpu - whether to utilize gpu to train
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('--arch', type=str, default='DenseNet', help='Model architecture to use for image classification')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--hidden_units', type=int, default=1000, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Utilize gpu to train')

    return parser.parse_args()

def get_command_args_predict():
    """
    Get argument parsed prediction
    
    Arguments:
       input - pathe ti image to predict
       checkpoint -saved model / checkpoint file
       category_names - class mapping
       top_k most likely classes - top k probablities
       gpu - Use Gpu
       
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', help='Input for path to file to predic')
    parser.add_argument('checkpoint_file', help='Input for saved checkpoint file')
    parser.add_argument('--category_names', type=str,
                        default='cat_to_name.json', help='Path to classes map file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of highest probabilities')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU (Cuda)')

    return parser.parse_args()

# Helpers used in the course (Part 3 - Training Neural Networks)
def showImage(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True

# Rubric: [Done] 11. Saving the model
def save_model(model, save_dir, hidden_layers, learning_rate, output_size, epochs, optimizer, data):
    checkpoint = {'architecture': model.__class__.__name__,
              'input_size': model.classifier[0].in_features,
              'hidden_layers': hidden_layers,
              'output_size': output_size,
              'learning_rate': learning_rate,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': data.class_to_idx}
    
        # Create a folder to save checkpoint if not already existed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Saved Checkpoint Format: {architecture}_{input_size}_{hidden_layers}_{output_size}_{epochs}_{learning_rate}_checkpoint.pth
    file_path = save_dir + '/{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(checkpoint['architecture'], checkpoint['input_size'], checkpoint['hidden_layers'], checkpoint['output_size'], checkpoint['epochs'], checkpoint['learning_rate'])
    
    #file_path = "densenet121_checkpoint.pth"
    
    torch.save(checkpoint, file_path)
    print("Checkpoint Saved!")
    
# Rubric: [Done] 12. Loading checkpoints
def load_model(filepath):
    # Use cuda if availble (Part 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load Model:{}".format(device))
    
    # See https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=lambda storage, location: storage)
    #checkpoint = torch.load(filepath, map_location=device)
    
    print(checkpoint)
    
    # Load the pre-trained model and rebuild our model
    model = select_network(checkpoint['architecture'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    print('Checkpoint loaded successfully!')
    
    return model, checkpoint

def map_labels(labels_path):
    with open(labels_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

# Extract from Notebook
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    img_tensor = preprocess(image)
    
    return img_tensor.numpy()

# Extract from Notebook
def load_data(data_directory):
    training_dir = data_directory + '/train'
    testing_dir = data_directory + '/test'
    validiation_dir = data_directory + '/valid'
    
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    testing_data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valididation_data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    training_data = datasets.ImageFolder(training_dir, transform=training_data_transforms)
    testing_data = datasets.ImageFolder(testing_dir, transform=testing_data_transforms)
    validation_data = datasets.ImageFolder(validiation_dir, transform=valididation_data_transforms)

    #Rubric: [Done] Data Batching
    batch_size = 64
    testing_batch_size = 32

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    trainingloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    testingloader = torch.utils.data.DataLoader(testing_data, batch_size=testing_batch_size)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=testing_batch_size)
    
    return training_data, testing_data, validation_data, trainingloader, testingloader, validationloader

def validation(model, loader, criterion):
    loss = 0
    accuracy = 0
    
    # test model with cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Deactivate drop-out
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    # Activate drop-out
    model.train()
    
    return loss, accuracy

def select_network(arch):
    ''' Selects the pretrained network that is passed to it
    Select from VGG, DenseNet or AlexNet.
    '''
    if arch == 'VGG16':
        model = models.vgg16(pretrained=True)        
    elif arch == 'DenseNet':
        model = models.densenet121(pretrained=True)
    elif arch == 'AlexNet':
        model = models.alexnet(pretrained = True)
    else:
        print("Please input a valid model. Select from VGG, DenseNet or AlexNet?")
        
    return model

def check_accuracy_on_test(model, testloader):  
    correct = 0
    total = 0
    image_count = 0
    
    # test model with cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # turn off drop-out
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on {} test images: {:.3f}%'.format(total, 100 * correct / total))
    
def setup_network(model, architecture, input_size, hidden_layers, output_size = 102, learning_rate = 0.001, drop_out=0.5):
      
    # Freeze parameters so we don't backprop through them (Part 8)
    for param in model.parameters():
        param.requires_grad = False
    
    # Build as sequential model
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_layers[0])),
                              ('relu1', nn.ReLU()),
                              ('drop_out1', nn.Dropout(p=drop_out)),
                              ('fc2', nn.Linear(hidden_layers[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    # Define lost function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    return model, optimizer, criterion

# Extract from Notebook
def train_network(model, trainingloader, validationloader, epochs, print_interval, criterion, optimizer, device):
    print("Training {} || epochs={}.".format(model.__class__.__name__, epochs))
    iterations = 0
    
    # Activate drop-out
    model.train()
    
    # Use cuda if availble (Part 8)
    # Rubric: Training with GPU - The training script allows users to choose training the model on a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Device:", device)
    
    if device == "cpu":
        print("Warning: GPU is not available, slow training...")
    
    # Set timer
    start_time = time()
    
    # Start loop
    for epoch in range(epochs):
        running_loss = 0
        
        # Turn drop-out on, just to make sure
        model.train()
        
        for index, (inputs, labels) in enumerate(trainingloader):
            iterations += 1
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Rubric: [Done] 7. Feedforward Classifier
            # Forward and backward passes (Part 3)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if iterations % print_interval == 0:
                print("Validating on Epoch:{}, Iteration:{}".format(epoch, iterations))
                # Rubric: [Done] 9. Validation Loss and Accuracy
                validation_loss, accuracy = validation(model, validationloader, criterion)
               
                # Calculate avg losses and accuracy
                running_loss = running_loss / len(trainingloader)
                validation_loss = validation_loss / len(validationloader)
                accuracy = accuracy / len(validationloader)
                
                #Rubric: Training validation log: The training loss, validation loss, and validation accuracy are printed out as a network trains
                print("Device={} - Epoch: {}/{} ... ".format(device, epoch + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_interval),
                      "Validation Loss: {:.3f}.. ".format(validation_loss / len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validationloader)))
                running_loss = 0

    tot_time = time() - start_time
    tot_time = strftime('%H:%M:%S', localtime(tot_time))
    print("\n** Total Training Runtime: ", tot_time)