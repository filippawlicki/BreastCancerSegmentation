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
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("../training_outputs/final_model.pth"))
model.eval()

root_dir = "../dataset"
splits = split_dataset(root_dir)
transforms = T.Compose([T.Resize((128, 128)), T.ToTensor()])

test_dataset = BreastUltrasoundDataset(
    splits["test"][0], splits["test"][1], transform=transforms, mask_transform=transforms
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create a subdirectory for saving the images
output_images_dir = '../predictions_output'
os.makedirs(output_images_dir, exist_ok=True)

import matplotlib.pyplot as plt
import torch
import random

def plot_samples(model, test_dataset, device):
    sample_indices = [52, 2, 66, 1, 73]

    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(10, 15))

    model.eval()
    dice_scores = []

    for i, idx in enumerate(sample_indices):
        inputs, masks = test_dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)
        masks = masks.unsqueeze(0).to(device)

        # Make a prediction
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # Apply threshold to get binary mask

        # Calculate Dice score
        intersection = torch.logical_and(preds.squeeze(), masks.squeeze()).sum()
        union = preds.sum() + masks.sum()
        dice = (2.0 * intersection) / (union + 1e-7)
        dice_scores.append(dice.item())

        # Convert tensors to NumPy arrays for visualization
        img = inputs.squeeze().cpu().numpy()
        gt_mask = masks.squeeze().cpu().numpy()
        pred_mask = preds.squeeze().cpu().numpy()

        # Plot images
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Sample {idx} - Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title(f"Prediction (Dice: {dice:.3f})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_images_dir, "Sample_Images.png"))
    plt.close()


# Function to save images on a single plot
def save_images(inputs, preds, masks, idx, output_dir):
    # Create a single plot with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # Input image
    axs[0].imshow(inputs.squeeze().cpu().numpy(), cmap="gray")
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

dice_score = 0

for idx in range(len(test_dataset)):
    inputs, masks = test_dataset[idx]
    inputs = inputs.unsqueeze(0).to(device)
    masks = masks.unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5

    # Compute Dice score
    intersection = torch.logical_and(preds.squeeze(), masks.squeeze()).sum()
    union = preds.sum() + masks.sum()
    dice = (2.0 * intersection) / (union + 1e-7)
    dice_score += dice.item()

    # Save the images in one plot
    save_images(inputs, preds, masks, idx, output_images_dir)


plot_samples(model, test_dataset, device)
print(f"Average Dice Score: {dice_score / len(test_dataset)}")
print(f"Images saved in {output_images_dir}")
