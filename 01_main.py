import importlib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import modules using importlib
data = importlib.import_module('02_data')
functions = importlib.import_module('03_functions')

# Load the dataset
get_lego_minifig_data = data.get_lego_minifig_data
clear_minifig_images_directory = data.clear_minifig_images_directory
image_blur = functions.image_blur



if __name__ == "__main__" :
  # Star Wars database
  star_wars_database = "rebrickable_sets_z4dis.csv"
  
  # Clear the minifig_images directory
  image_dir = "minifig_images"
  #clear_minifig_images_directory(image_dir)
  
  
  
  # Load the dataset
  minifig_database = "minifigs.csv"
  #get_lego_minifig_data(minifig_database, star_wars_database, image_dir="minifig_images")

  # Show augmented image
  image_path = os.path.join(image_dir, "Kylo_Ren748.jpg")  # Example image path
  image = plt.imread(image_path) 
  blurred_image = image_blur(image, blur_factor=0.6)  # Apply blur with a factor of what you want

  

