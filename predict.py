# Wesley Christelis
# Date: 05/01/2019
# Use already trained netwrok to predict a flower from an input image

'''
Basic usage: python predict.py /path/to/image checkpoint
Options:
    Return top n most likely classes: python predict.py input_image checkpoint_file --top_k 3
    Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    Use GPU for inference: python predict.py input checkpoint --gpu

Examples Tested:
    python predict.py flowers/test/1/image_06743.jpg densenet121_checkpoint.pth --top_k 3
    python predict.py flowers/test/1/image_06752.jpg densenet121_checkpoint.pth --category_names cat_to_name.json --gpu
    python predict.py flowers/test/1/image_06764.jpg densenet121_checkpoint.pth --gpu
    python predict.py flowers/test/1/image_06743.jpg densenet121_checkpoint.pth --top_k 3 --gpu
    python predict.py flowers/test/1/image_06752.jpg densenet121_checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
    python predict.py flowers/test/1/image_06752.jpg densenet121_checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
    python predict.py /flowers/test/102/image_08004.jpg densenet121_checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
    python predict.py /flowers/test/21/image_06805.jpg densenet121_checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
'''
# Imports python modules

#https://docs.python.org/3/library/argparse.html
import argparse
import torch
import torch.nn.functional as nnFunctional
import torch
from torch import nn
from torch import optim

import terminal_utils as  utils

#Rubric: Predicting classes: The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
def main():
    print("Prediction Start !")
    print(utils.test_helper())
    
    arguments_passed = utils.get_command_args_predict()

    image_path = arguments_passed.input_image
    checkpoint_file = arguments_passed.checkpoint_file
    categories = arguments_passed.category_names
    top_k = arguments_passed.top_k
    use_gpu = arguments_passed.gpu

    if(arguments_passed.gpu):
        if torch.cuda.is_available():
            device = 'gpu'
            print('Cuda activated')
        else:
            device = 'cpu'
            print('Cude deactivated')
    else:
        device = 'cpu'
        print('Using CPU for calculations')
    
     # Load checkpoint and rebuild our model
    saved_model, checkpoint = utils.load_model(checkpoint_file)
    print('Model loaded !!!')

    # Get index to class map
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Predict input image
    probabilities, classes = predict(image_path, saved_model, idx_to_class, top_k, device)

    #Rubric: Displaying class names: The predict.py script allows users to load a JSON file that maps the class values to other category names
    category_name = utils.map_labels(categories)
    names = [category_name[obj] for obj in classes]

    print('Classes          :', names)
    print('Probabilities (%):', [float(round(p * 100.0, 2)) for p in probabilities])

    print("Top most likely class:", names[0])

#Extract from notebook
def predict(image_path, model, idx_to_class, top_k=5, device='cpu'):
    '''
    Predict the class of Image
    '''
    print("Predicting the top {} classes; model {}; using device={}.".format(top_k, model.__class__.__name__, device))

    # Load Image
    processed_image = utils.process_image(image_path).squeeze()

    # Deactivate drop out
    model.eval()
    
    
    #Rubric: Predicting with GPU - The predict.py script allows users to use the GPU to calculate the predictions
    if device == 'gpu':
        model = model.cuda()

    with torch.no_grad():
        if device == 'gpu':
            output = model(torch.from_numpy(
                processed_image).float().cuda().unsqueeze_(0))
        else:
            output = model(torch.from_numpy(
                processed_image).float().unsqueeze_(0))

    # Calculate the class probabilities (softmax) for image
    ps = nnFunctional.softmax(output, dim=1)

    #Rubric: Top K classes: The predict.py script allows users to print out the top K classes along with associated probabilities
    top = torch.topk(ps, top_k)

    probabilities = top[0][0].cpu().numpy()
    classes = [idx_to_class[i] for i in top[1][0].cpu().numpy()]

    return probabilities, classes

# Call to main function to run the program
if __name__ == "__main__":
    main()