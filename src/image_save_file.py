import os
from datetime import datetime as date_time
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def count_file(directory_path_temp):
    unique_id_temp = 0
    existing_files = len([f for f in os.listdir(directory_path_temp) if (f.endswith(".png") or f.endswith(".jpg")) and (os.path.isfile(os.path.join(directory_path_temp, f)))])
    unique_id_temp = existing_files + 1
    return unique_id_temp

def count_folders(directory_path_temp, new_folder):
    unique_id_temp = 0
    existing_folders = [
        int(d.split('_')[0]) for d in os.listdir(directory_path_temp) if (os.path.isdir(os.path.join(directory_path_temp, d)) and re.search(r'^\d+', d))
    ]
    if existing_folders:
        unique_id_temp = max(existing_folders)
        if new_folder:
            unique_id_temp = unique_id_temp + 1    
    else:
        unique_id_temp = 1
    return str(unique_id_temp)

def add_metadata_file(file_path, txt_file_data_file):
    targetImage = Image.open(file_path)
    metadata = PngInfo()
    metadata.add_text("parameters", txt_file_data_file)
    targetImage.save(file_path, pnginfo=metadata)


def save_file(image_file, txt_file_data_file):
    file_path = ""
    if image_file != "":
        current_datetime = date_time.now()
        current_date = current_datetime.strftime(f"%Y_%m_%d")
        current_time = current_datetime.strftime(f"%H_%M_%S")
        if not os.path.exists("./image"):
            os.makedirs("./image")
        if not os.path.exists("./image/" + current_date):
            os.makedirs("./image/" + current_date)
        directory_path = f"./image/{current_date}"
        print(f"Directory:{directory_path}")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        unique_id = count_file(directory_path)
        file_name = f"{unique_id}_{current_time}.png"
        file_path = f"{directory_path}/{file_name}"
        image_file.save(file_path)
        add_metadata_file(file_path, txt_file_data_file)
    return file_path
