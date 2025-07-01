import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
from torch.utils.data import DataLoader, Subset
import torchvision.utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset and resize to 128x128
transform_load = transforms.Compose([
    transforms.ToTensor(),             # Convert PIL Image to PyTorch Tensor (0-1 range)
    transforms.Resize(128, antialias=True), # Resize to 128x128, anti-aliasing for better quality
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_load)

def preview_data():
    preview_dataloader = DataLoader(mnist_dataset, batch_size=6, shuffle=True)
    # Get the first batch of images
    batch = next(iter(preview_dataloader))
    # Extract the images from the batch
    images = batch[0]

    # Plot the images as a grid
    plt.figure(figsize=(8, 8))
    for i, img in enumerate(images):
        plt.subplot(1, 6, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

def diffusion_model():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,    # number of steps
        sampling_timesteps = 250 
    )

    trainer = Trainer(
        diffusion,
        "./mnist_resized_images/",
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True              # whether to calculate fid during training
    )
    trainer.train()

    sampled_images = diffusion.sample(batch_size = 4)
    sampled_images.shape # (4, 3, 128, 128)

if __name__ == '__main__':
    diffusion_model()