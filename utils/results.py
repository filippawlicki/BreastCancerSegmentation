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

# Select 5 random samples from the dataset
random_indices = random.sample(range(len(test_dataset)), 5)

# Function to display images
def show_images(inputs, preds, masks, idx):
    plt.figure(figsize=(15, 5))

    # Display Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(inputs[idx].cpu().numpy().transpose(1, 2, 0))
    plt.title("Input Image")
    plt.axis("off")

    # Display Prediction Mask
    plt.subplot(1, 3, 2)
    plt.imshow(preds[idx].cpu().numpy().squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Display Ground Truth Mask
    plt.subplot(1, 3, 3)
    plt.imshow(masks[idx].cpu().numpy().squeeze(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.show()

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

    # Display the images and masks
    show_images(inputs, preds, masks, 0)
