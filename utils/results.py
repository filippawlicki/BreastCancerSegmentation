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

# Load the dataset and DataLoader
root_dir = "../dataset"  # Path to your dataset
splits = split_dataset(root_dir)
transforms = T.Compose([T.Resize((128, 128)), T.ToTensor()])

test_dataset = BreastUltrasoundDataset(
    splits["test"][0], splits["test"][1], transform=transforms, mask_transform=transforms
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create a subdirectory for saving the images
output_images_dir = '../predictions_output'
os.makedirs(output_images_dir, exist_ok=True)

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

# Test the model on 5 random samples
for inputs, masks in test_dataset:
    # Get the image and its ground truth mask
    #inputs, masks = test_dataset[idx]
    inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
    masks = masks.unsqueeze(0).to(device)  # Add batch dimension

    # Get the index of the sample
    idx = random.randint(0, len(test_dataset) - 1)

    # Make a prediction
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5  # Apply threshold to get binary mask

    # Calculate the Dice score
    intersection = torch.logical_and(preds.squeeze(), masks.squeeze()).sum()
    union = preds.sum() + masks.sum()
    dice = (2.0 * intersection) / (union + 1e-7)
    dice_score += dice.item()

    # Save the images in one plot
    save_images(inputs, preds, masks, idx, output_images_dir)


print(f"Average Dice Score: {dice_score / len(test_dataset)}")
print(f"Images saved in {output_images_dir}")
