import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# Load MNIST dataset and resize to 128x128
transform_load = transforms.Compose([
    transforms.ToTensor(),             # Convert PIL Image to PyTorch Tensor (0-1 range)
    transforms.Resize(128, antialias=True), # Resize to 128x128, anti-aliasing for better quality
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_load)

# Define the root folder where you want to save the images
output_root_folder = './mnist_resized_images'

# Create the root output folder if it doesn't exist
os.makedirs(output_root_folder, exist_ok=True)

print(f"Extracting {len(mnist_dataset)} resized images from MNIST dataset...")

# Iterate through the dataset
for i, (image_tensor, label) in enumerate(mnist_dataset):
    # Convert the PyTorch Tensor back to a PIL Image
    # MNIST images are grayscale, so we convert them to 'L' mode (luminance)
    image_pil = transforms.ToPILImage()(image_tensor.squeeze(0)) # .squeeze(0) to remove the channel dimension for grayscale

    # Create subfolders for each digit (0-9) if they don't exist
    label_folder = os.path.join(output_root_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # Define the filename for the image
    image_filename = os.path.join(label_folder, f'{i:05d}.png')

    # Save the image
    image_pil.save(image_filename)

    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1}/{len(mnist_dataset)} images.")

print("Image extraction complete!")