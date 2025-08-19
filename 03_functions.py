# Créer une fonction qui va aléatoirement enlever des pixels de chaque image afin de pouvoir entrainer le modèle. Utiliser l'exemple en bookmark sur kaggle sur minifigs lego.

import os
import math
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import albumentations as A
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2 as tf_mobilenet_v2
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import models as tf_models
from tensorflow.keras import callbacks as tf_callbacks
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_model_selection

def image_blur(image, blur_factor):
    """
    Apply a Gaussian blur to the image.
    
    Parameters:
    - image: Input image as a NumPy array.
    - blur_factor: Factor by which to blur the image (0 to 1).
    
    Returns:
    - Blurred image as a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    if not (0 <= blur_factor <= 0.65):
        raise ValueError("Blur factor must be between 0 and 1.")
    
    kernel_size = int(blur_factor * 10) | 1  # Ensure kernel size is odd
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    plt.imshow(blurred_image)
    plt.show()
    return blurred_image