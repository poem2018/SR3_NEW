from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

######check resolution############
def check_image_resolutions(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                if img.size[0]!=1024:
                    print("!!!")
                # print(f"{filename}: {img.size}")
                # break

folder_path =  "../dataset/finetune_8x/sr_128_1024" # Replace with your folder path
check_image_resolutions(folder_path)




# #############remove borke image###############
# def remove_broken_images(directory):
#     removed_count = 0
#     # Iterate through all files in the directory
#     count=0
#     for file_path in Path(directory).glob('*'):
#         if count%1000==0:
#             print(count) 
#         count+=1
#         if file_path.is_file():
#             try:
#                 # Try to open the image
#                 with Image.open(file_path) as img:
#                     # If successful, check if the image is loaded properly
#                     img.verify()
#             except (IOError, SyntaxError) as e:
#                 # If an error occurs, remove the file
#                 print(f"Removing broken image: {file_path}")
#                 os.remove(file_path)
#                 removed_count += 1
#     return removed_count

# # Specify the directory path
# dir_path = './dataset/Celeba_2x_512_1024_256/sr_256_512'

# # Remove broken images and get the count of removed files
# removed_files = remove_broken_images(dir_path)
# print(f"Total removed files: {removed_files}")




# ############pixel-level variance##############
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = Image.open(os.path.join(folder, filename))
#         if img is not None:
#             images.append(np.asarray(img))
#     return images

# def calculate_pixel_variance(images):
#     # Stack images along a new dimension
#     stack = np.stack(images, axis=0)
#     # Calculate variance along the stack
#     return np.var(stack, axis=0)

# def visualize_variance(variance):
#     plt.imshow(variance, cmap='hot')
#     plt.colorbar()
#     plt.title("Pixel-wise Variance")
#     plt.show()

# # Folder containing images
# folder = '/path/to/your/image/folder'

# # Load images
# images = load_images_from_folder(folder)
# variance = calculate_pixel_variance(images)
# visualize_variance(variance)
