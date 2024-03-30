import json
import os
import pandas as pd
import numpy as np
from PIL import Image

def read_json(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data
def save_json(path, data, override=True):
    if override:
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        old_data = read_json(path)
        old_data.append(data)
        with open(path, 'w') as json_file:
            json.dump(old_data, json_file, indent=4)
            
def read_config(path, featurelist):
    config = read_json(path)
    results = {}
    for feature in featurelist:
        results[feature] = config[feature]
    return results
            
def mkdir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory checkpoint is created!")

def read_csv(path):
    if 'gz' in path:
        compression = 'gzip'
    else:
        compression = None
    df = pd.read_csv(path, compression=compression,header=0)
    return df

def next_path(dir, extension):
    mkdir(dir)
    file_prefix = "loop_"
    existing_files = [file for file in os.listdir(dir) if file.startswith(file_prefix)]
    num_existing_files = len(existing_files)
    path = f"{dir}/{file_prefix}{num_existing_files}{extension}"
    return path

def cur_path(dir, extension):
    mkdir(dir)
    file_prefix = "loop_"
    existing_files = [file for file in os.listdir(dir) if file.startswith(file_prefix)]
    num_existing_files = len(existing_files)
    path = f"{dir}/{file_prefix}{num_existing_files-1}{extension}"
    return path


def delete_folder(folder_path):
    try:
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")

def save_PIL(path, image):
    image.save(path)

def convert_12_to_8(image):
    scaled_data = (image / 4095 * 255).astype(np.uint8)

    # Reshape the data to the desired shape
    image_data = scaled_data.reshape((271, 271))

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_data)

    return image