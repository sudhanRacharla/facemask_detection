
# Face Mask detection using CNN

In this project, we'll develop an ML model that can recognise people wearing facemasks.

# Detailed Overview

For image predictions, CNN will be used because it is effective at extracting features. You must construct your own dataset for this duty, which includes dividing up photographs of persons wearing masks and those without them into two distinct files. 

The model is constructed using the CNN method. The model trainig code is located in the trainig.py file. The trained models are included. The code for testing a model with images is in the file img_predict.py. The final file containing the execution code is called exe.py. We utilise opencv to build a live detection system in which a web camera will collect the input and a model will classify it.

info.txt file has the information about the model layers, activation functions.
## Deployment

To deploy this project follow the steps:

1. Clone this repository
2. Change the paths of harcascade_frontalface_default.xml and model.h5 files in the exe.py file
3. Install necessary packages
4. Execute the exe.py file

## Author

[Sudhan Jee](https://github.com/sudhanRacharla)

