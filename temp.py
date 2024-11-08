import os
import shutil
import json

dataset_root = r'C:\Users\91875\Downloads\pix3d'
organized_root = r'C:\Users\91875\Downloads\Pix3D_organized'
json_file_path = os.path.join(dataset_root, 'pix3d.json')

with open(json_file_path, 'r') as f:
    data = json.load(f)

if not os.path.exists(organized_root):
    os.makedirs(organized_root)

for entry in data:
    image_file = entry.get('img')    
    model_file = entry.get('model')  
    mask_file = entry.get('mask')     
    class_name = entry.get('category') 

    full_image_path = os.path.join(dataset_root, image_file)
    full_model_path = os.path.join(dataset_root, model_file)
    full_mask_path = os.path.join(dataset_root, mask_file)

    if not os.path.exists(full_image_path) or not os.path.exists(full_model_path) or not os.path.exists(full_mask_path):
        continue

    class_folder = os.path.join(organized_root, class_name)
    os.makedirs(class_folder, exist_ok=True)

    model_folder_name = os.path.basename(os.path.dirname(model_file))
    instance_folder = os.path.join(class_folder, model_folder_name)
    images_folder = os.path.join(instance_folder, 'images')
    models_folder = os.path.join(instance_folder, 'models')
    masks_folder = os.path.join(instance_folder, 'masks')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    shutil.copy(full_image_path, images_folder)    
    shutil.copy(full_model_path, models_folder)    
    shutil.copy(full_mask_path, masks_folder)      

    print(f"Copied {full_image_path}, {full_model_path}, and {full_mask_path} to {instance_folder}")

print("Organizing completed.")
