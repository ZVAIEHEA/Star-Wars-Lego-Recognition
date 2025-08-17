import pandas as pd
import os
import requests

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

          with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
              file.write(chunk)
          print(f"Downloaded {file_name}")
        except requests.exceptions.RequestException as e:
          print(f"Error downloading {img_url}: {e}")