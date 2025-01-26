import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms as T
from utils.dataset import BreastUltrasoundDataset, split_dataset
from models.unet_model import UNet
import os
from torch.utils.data import DataLoader

# Load the trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("../training_outputs/final_model.pth"))
model.eval()

# Load the dataset and DataLoader
root_dir = "../dataset"  # Path to your dataset
splits = split_dataset(root_dir)
transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])

test_dataset = BreastUltrasoundDataset(
    splits["test"][0], splits["test"][1], transform=transforms, mask_transform=transforms
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create a subdirectory for saving the images
output_images_dir = './predictions_output'
os.makedirs(output_images_dir, exist_ok=True)

# Select 5 random samples from the dataset
random_indices = random.sample(range(len(test_dataset)), 5)

# Function to save images on a single plot
def save_images(inputs, preds, masks, idx, output_dir):
    # Create a single plot with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # Input image
    axs[0].imshow(inputs.squeeze().cpu().numpy().transpose(1, 2, 0))
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    # Predicted mask
    axs[1].imshow(preds.squeeze().cpu().numpy(), cmap="gray")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    # Ground truth mask
    axs[2].imshow(masks.squeeze().cpu().numpy(), cmap="gray")
    axs[2].set_title("Ground Truth Mask")
    axs[2].axis("off")

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"))
    plt.close()

# Test the model on 5 random samples
for idx in random_indices:
    # Get the image and its ground truth mask
    inputs, masks = test_dataset[idx]
    inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
    masks = masks.unsqueeze(0).to(device)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5  # Apply threshold to get binary mask

    # Save the images in one plot
    save_images(inputs, preds, masks, idx, output_images_dir)

print(f"Images saved in {output_images_dir}")
