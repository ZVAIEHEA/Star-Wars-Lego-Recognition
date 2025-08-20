import pandas as pd
import os
import requests
from PIL import Image
import importlib
import random
import matplotlib.pyplot as plt

# Import modules using importlib
functions = importlib.import_module('03_functions')

image_blur = functions.image_blur


def clear_minifig_images_directory(image_dir="minifig_images"):
  if os.path.exists(image_dir):
    for file_name in os.listdir(image_dir):
      file_path = os.path.join(image_dir, file_name)
      os.remove(file_path)
      print(f"Removed {file_path}")
  else:
    os.mkdir(image_dir)
    print(f"Created {image_dir} directory")

def get_lego_minifig_data(minifig_database, star_wars_database ,image_dir="minifig_images"):

  # Load the dataset
  df_image = pd.read_csv(minifig_database)
  df_star_wars = pd.read_csv(star_wars_database)

  # Remove the ban characters from the name column
  df_image['name'] = df_image['name'].str.replace(r'[^\w\s]', '', regex=True)

  # Add the image from the dataframe to the directory
  # Merge the two dataframes

  minifig_name = '' # Initialize minifig_name with an empty string

  # Download and save images
  while minifig_name != 'stop' :
    df = pd.merge(df_star_wars, df_image, left_on='Set Number', right_on='fig_num', how='left')
    minifig_name = input("Enter the name of the minifig you want to download: ")
    df = df[df['name'].str.contains(minifig_name, case=False, na=False)]
    print(df)
    for index, row in df.iterrows():
      if pd.notnull(row['img_url']):
        img_url = row['img_url']
        minifig_name = minifig_name.replace(" ", "_")
        file_name = os.path.join(image_dir, f"{minifig_name}{index}.jpg")  # Use 'Set Number' for filename

        try:
          response = requests.get(img_url, stream=True)
          response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
          # Save the image temporarily
          temp_file = os.path.join(image_dir, f"temp_{minifig_name}{index}.jpg")
          with open(temp_file, 'wb') as file:
              for chunk in response.iter_content(chunk_size=8192):
                  file.write(chunk)

          # Open the image with Pillow, resize it, and save it as 256x256
          with Image.open(temp_file) as img:
              img = img.convert("RGB")  # Ensure the image is in RGB format
              img = img.resize((256, 256))  # Resize to 256x256
              img.save(file_name)  # Save the resized image
          os.remove(temp_file)  # Remove the temporary file
          print(f"Downloaded and resized {file_name}")

        except requests.exceptions.RequestException as e:
          print(f"Error downloading {img_url}: {e}")
        except Exception as e:
          print(f"Error processing {file_name}: {e}")


def create_training_and_test_datasets(image_dir="minifig_images"):
  
  images = []
  train_images = []
  test_images = []

  if os.path.exists("train_images"):
    print("train_images directory already exists")
  else:
    os.makedirs("train_images")
    print("Created train_images directory")

  if os.path.exists("test_images"):
    print("test_images directory already exists")
  else:
    os.makedirs("test_images")
    print("Created test_images directory")
  
  



