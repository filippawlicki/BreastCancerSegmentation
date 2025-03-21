import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import to_tensor, to_pil_image

class BreastUltrasoundDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_paths = self.mask_paths[idx]

        # Load image
        image = Image.open(img_path).convert("L")

        # Combine multiple masks
        masks = [Image.open(mask_path).convert("L") for mask_path in mask_paths]
        combined_mask = None
        for mask in masks:
            mask_tensor = to_tensor(mask)  # Convert mask to a tensor
            if combined_mask is None:
                combined_mask = mask_tensor
            else:
                combined_mask += mask_tensor

        # Clamp combined_mask values to [0, 1] (in case of overlaps)
        combined_mask = combined_mask.clamp(0, 1)

        combined_mask = to_pil_image(combined_mask)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            combined_mask = self.mask_transform(combined_mask)

        return image, combined_mask

def split_dataset(root_dir, val_size=0.2, test_size=0.1, random_seed=42):
    random.seed(random_seed)
    all_images = []
    all_masks = []

    for label in ["normal", "malignant", "benign"]:
        image_dir = os.path.join(root_dir, label)
        for img_path in glob.glob(os.path.join(image_dir, "*.png")):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_pattern = os.path.join(image_dir, f"{base_name}_mask*.png")
            masks = glob.glob(mask_pattern)

            if masks:
                all_images.append(img_path)
                all_masks.append(masks)

    # Split into train, val, and test
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        all_images, all_masks, test_size=(val_size + test_size), random_state=random_seed
    )
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=(test_size / (val_size + test_size)), random_state=random_seed
    )

    return {
        "train": (train_images, train_masks),
        "val": (val_images, val_masks),
        "test": (test_images, test_masks),
    }
