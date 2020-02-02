# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Create your conda env using yaml
- create environment (yaml is in project directory)
     conda env create -f ai_environment_final_project.yml

## __Project Structure__
project
|----assets
|    |    *.png
|----scripts
|    |
|----checkpoints
|    | DenseNet_1024_[1024]_102_1_0.001_checkpoint.pyh
|    | DenseNet_1024_[1024]_102_5_0.001_checkpoint.pyh    
|    cat_to_name.json
|    LICENCE
|    predict.py
|    README.md
|    train.py
|    terminal_utils.py

# Rubric
##### Files submitted

Submission Files: The submission includes all required files.

##### Part 1 - Development Notebook
Criteria | Specification

[Done] 1. __*Package Imports*__	| All the necessary packages and modules are imported in the __first__ cell of the notebook

[Done] 2. __*Training data augmentation*__ |	torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

[Done] 3. __*Data normalization*__ |	The training, validation, and testing data is appropriately cropped and normalized

[Done] 4. __*Data loading*__ | The data for each set (__train__, __validation__, __test__) is loaded with torchvision's ImageFolder

[Done] 5. __*Data batching*__ | The data for each set is loaded with torchvision's DataLoader

[Done] 6. __*Pretrained Network*__ | A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are __frozen__

[Done] 7. __*Feedforward Classifier*__ | A new feedforward network is defined for use as a classifier using the features as input

8. __*Training the network*__ | The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static

[Done] 9. __*Validation Loss and Accuracy*__ | During training, the validation loss and accuracy are displayed

[Done] 10. __*Testing Accuracy*__ | The network's accuracy is measured on the test data

[Done] 11. __*Saving the model*__ | The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

[Done] 12. __*Loading checkpoints*__ | There is a function that successfully loads a checkpoint and rebuilds the model

[Done] 13. __*Image Processing*__ | The process_image function successfully converts a PIL image into an object that can be used as input to a trained model

[Done] 14. __*Class Prediction*__	| The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

[Done] 15. __*Sanity Checking with matplotlib*__ | A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

##### Part 2 - Command Line Application
Criteria | Specification

[Done] 1. __*Training a network*__ | train.py successfully trains a new network on a dataset of images

[Done] 2. __*Training validation log*__ | The training loss, validation loss, and validation accuracy are printed out as a network trains

[Done] 3. __*Model architecture*__ |	The training script allows users to choose from at least two different architectures available from torchvision.models

[Done] 4. __*Model hyperparameters*__ |	The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs

[Done] 5. __*Training with GPU*__ |	The training script allows users to choose training the model on a GPU

[Done] 6. __*Predicting classes*__ |	The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability

[Done] 5. __*Top K classes*__ |	The predict.py script allows users to print out the top K classes along with associated probabilities

[Done] 6. __*Displaying class names*__ |	The predict.py script allows users to load a JSON file that maps the class values to other category names

[Done] 7. __*Predicting with GPU*__ |	The predict.py script allows users to use the GPU to calculate the predictions


