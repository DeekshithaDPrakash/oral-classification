# Processing Input and Output Data
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import cv2

LABELS=['oral', 'nonoral']

def get_image(arg_parser):
    '''Returns a Pillow Image given the uploaded image.'''
    args = arg_parser.parse_args()
    image_file = args.image  # reading args from file
    return Image.open(image_file)  # open the image


def preprocess_image(image):
    """Converts a PIL.Image into a Tensor of the 
    right dimensions 
    """
    size=(150,150)
    image_data=ImageOps.fit(image, size)
    image_data= np.asarray(image_data)
    new_image= cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    resized_image= cv2.resize(new_image, (256,256))
    final_image=np.reshape(resized_image, [1,256,256,3])
    
    return final_image


def predict_oral(model, image):
    """Returns the most likely class for the image 
    according to the output of the model.

    Parameters: model and image

    Source: https://tinyurl.com/dzav422a

    Returns: dict: the label-ORAL or NONORAL and the models confidence-percentage of correct prediction associated thereof it
                   are included as fields
    """
    prediction_probabilities = model.predict(image)
    # get the prediction label
    index_highest_proba = np.argmax(prediction_probabilities)
    label = str(LABELS[index_highest_proba])
    # get the prediction probability
    confidence = float(100*np.max(prediction_probabilities))
    # return the output as a JSON string
    output = {
         "success": True, 
         "confidence": confidence
    }
    return output
