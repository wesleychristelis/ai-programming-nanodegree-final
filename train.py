# Wesley Christelis
# Date: 05/01/2019
# Train model on a specified dataset / archhitecture

'''
Basic usage: python train.py data_directory

Trains netwwork and outputs training loss, validation loss, and validation accuracy as it trains

Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu

Examples
python train.py flowers --gpu
python train.py flowers --arch "VGG" --save_dir checkpoints --learning_rate 0.002 --hidden_units 1024 --epochs 1 --gpu
python train.py flowers --arch "DenseNet" --save_dir checkpoints --learning_rate 0.001 --hidden_units 1024 --epochs 5 --gpu
'''

# Imports python modules
import argparse
from PIL import Image
import torch
from torch import nn
from torch import optim
import terminal_utils as utils

#Rubric: Training a network: train.py successfully trains a new network on a dataset of images
def main():
    print("Training Started !!!")
    
    arguments_passed = utils.get_command_args_train()
    
    data_dir = arguments_passed.data_dir
    arch = arguments_passed.arch
    save_dir = arguments_passed.save_dir
    device = 'gpu' if arguments_passed.gpu else 'cpu'
    
    # Rubric: Model hyperparameters - The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
    epochs = arguments_passed.epochs
    hidden_layers = [arguments_passed.hidden_units]
    output_size = 102
    learning_rate = arguments_passed.learning_rate
    drop_out = 0.5
    
    data_dir = arguments_passed.data_dir
    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'
    
     # Check arch
    # Rubric: Model architecture: The training script allows users to choose from at least two different architectures available from torchvision.models
    if arch not in ["VGG", "DenseNet"]:
        raise Exception(arch + " Unsupported Architecture! Please use 'VGG' or 'DenseNet' for --arch argument switch.")

    # Load data
    training_data, testing_data, validation_data, trainingloader, testingloader, validationloader = utils.load_data(data_dir)

    # Select model to train on
    selected_network_model = utils.select_network(arch)
    
    # Only Support 2
    if (selected_network_model.__class__.__name__ == "DenseNet"):
        input_size = selected_network_model.classifier.in_features
    elif (selected_network_model.__class__.__name__ == "VGG"):
        input_size = selected_network_model.classifier[0].in_features
    else:
        raise Exception("Architecture not supported!")
    
    # Build model
    model, optimizer, criterion = utils.setup_network(selected_network_model, arch, input_size, hidden_layers, output_size, learning_rate, drop_out)
   
    print('Training with hyperparams: epochs={}||input_size={}||hidden_layers={}||output_size={}||learning_rate={}||drop_p={}'.format(epochs, input_size, hidden_layers, output_size, learning_rate, drop_out))
    
    # Train our model
    print_interval = 25
    utils.train_network(model, trainingloader, validationloader, epochs, print_interval, criterion, optimizer, device)
    utils.check_accuracy_on_test(model, testingloader)
    
    # Save the checkpoint
    #def save_model(model, save_dir, hidden_layers, learning_rate, output_size, epochs, optimizer, data):
    utils.save_model(model, save_dir, hidden_layers, learning_rate, output_size, epochs, optimizer, training_data )
    
    print('Training Complete!')

# Call to main function to run the program
if __name__ == "__main__":
    main()
