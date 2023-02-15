import torch
import os
from PIL import Image
import concurrent.futures

# Load the face detection model from Torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt', force_reload=True)

# Set the detection threshold
model.conf = 0.9

# Define the input and output folders
input_folder = '.'
output_folder = 'output'
output_folder2 = 'output2'

# Create the output folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)

def process_image(filename):
    if filename.endswith('.png'):
        print(f"Processing image: {filename}")

        # Load the image
        image = Image.open(os.path.join(input_folder, filename))
        print(f"Loaded image: {filename}")

        # Use Torch to detect faces in the image
        results = model(image)
        faces = results.xyxy[0]
        print(f"Detected faces: {faces}")

        # If there is only one face detected, move the image to the output folder
        if len(faces) == 1:
            os.rename(os.path.join(input_folder, filename), os.path.join(output_folder, filename))
            print(f"Moved image to output folder: {filename}")

        # If there are multiple faces detected, move the image to the output2 folder
        elif len(faces) > 1:
            os.rename(os.path.join(input_folder, filename), os.path.join(output_folder2, filename))
            print(f"Moved image to output2 folder: {filename}")

# Create a list of filenames in the input folder
filenames = [filename for filename in os.listdir(input_folder) if filename.endswith('.png')]

# Use a ThreadPoolExecutor to process the images in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_image, filenames)
