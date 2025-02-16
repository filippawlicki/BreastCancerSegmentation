import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from utils.dataset import BreastUltrasoundDataset, split_dataset
from models.unet_model import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

import warnings

warnings.filterwarnings("ignore")

def compute_metrics(preds, labels):
    # Flatten tensors
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    # Convert logits to probabilities using sigmoid
    preds_flat = torch.sigmoid(preds_flat)

    # Convert labels to binary (0 or 1)
    labels_flat = (labels_flat > 0.5).float()  # This ensures binary labels (0 or 1)

    # Compute Dice Score
    intersection = (preds_flat * labels_flat).sum()
    dice_score = (2. * intersection + 1e-6) / (preds_flat.sum() + labels_flat.sum() + 1e-6)

    # Compute IoU Score
    iou = intersection / (preds_flat.sum() + labels_flat.sum() - intersection + 1e-6)

    # Compute Recall (Sensitivity)
    recall = intersection / (labels_flat.sum() + 1e-6)

    # Compute Precision
    precision = intersection / (preds_flat.sum() + 1e-6)

    # Compute Global Accuracy (check if each pixel is equal, then average over the entire mask)
    binary_preds_flat = preds_flat > 0.5
    global_accuracy = (binary_preds_flat == labels_flat).float().mean().item()

    # Compute AUC-ROC Score (requires probabilities)
    auc_roc = roc_auc_score(labels_flat.cpu().numpy(), preds_flat.cpu().numpy())

    metrics = {
        "dice_score": dice_score.item(),
        "iou_score": iou.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "global_accuracy": global_accuracy,
        "auc_roc": auc_roc
    }

    return metrics



def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, output_dir):
    """
    Train the model with progress tracking for each epoch and additional metrics logging.
    """
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_dice_score": [],
        "val_dice_score": [],
        "train_iou_score": [],
        "val_iou_score": [],
        "train_recall": [],
        "val_recall": [],
        "train_precision": [],
        "val_precision": [],
        "train_global_accuracy": [],
        "val_global_accuracy": [],
        "train_auc_roc": [],
        "val_auc_roc": []
    }
    best_metric = -float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total_samples = 0
            epoch_metrics = {
                "dice_score": 0,
                "iou_score": 0,
                "recall": 0,
                "precision": 0,
                "global_accuracy": 0,
                "auc_roc": 0
            }

            # Add a progress bar for this phase
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Progress", unit="batch", mininterval=1)

            for inputs, masks in progress_bar:
                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Convert output to binary (since it's a segmentation problem)
                preds = torch.sigmoid(outputs) > 0.5  # Convert to binary mask

                # Compute metrics
                batch_metrics = compute_metrics(preds, masks)
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

                # Optionally update progress bar with current loss
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / total_samples
            print(f"\n{phase.capitalize()} Loss: {epoch_loss:.4f}")

            # Log metrics
            metrics[f"{phase}_loss"].append(epoch_loss)
            print("\n")
            for key, value in epoch_metrics.items():
                epoch_metrics[key] /= len(dataloaders[phase])  # Average over the batch
                print(f"{phase.capitalize()} {key.capitalize()}: {epoch_metrics[key]:.4f}")
                metrics[f"{phase}_{key}"].append(epoch_metrics[key])

            # Save best model based on Dice Score (or any other metric)
            if phase == "val" and epoch_metrics["dice_score"] > best_metric:
                best_metric = epoch_metrics["dice_score"]
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"\nBest model saved with Dice Score: {best_metric:.4f}")

        print()

    # Save final metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot and save metrics graph
    plot_metrics(metrics, output_dir)

    return model


def plot_metrics(metrics, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_graph.png"))
    plt.close()

    # Plot the segmentation metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_dice_score"], label="Train Dice Score")
    plt.plot(metrics["val_dice_score"], label="Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "dice_score_graph.png"))
    plt.close()

def log_training_details(output_dir, params):
    with open(os.path.join(output_dir, "training_log.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    # Paths and Hyperparameters
    root_dir = "./dataset"  # Path to your dataset
    output_dir = "./training_outputs"  # Output directory for saving models and metrics
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    splits = split_dataset(root_dir)
    transforms = T.Compose([T.Resize((128, 128)), T.ToTensor()])

    train_dataset = BreastUltrasoundDataset(
        splits["train"][0], splits["train"][1], transform=transforms, mask_transform=transforms
    )
    val_dataset = BreastUltrasoundDataset(
        splits["val"][0], splits["val"][1], transform=transforms, mask_transform=transforms
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8),
    }

    # Initialize model, loss, and optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Log training details
    training_params = {
        "Dataset Path": root_dir,
        "Output Directory": output_dir,
        "Number of Epochs": num_epochs,
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "Device": device.type,
    }
    log_training_details(output_dir, training_params)

    # Train the model
    print("Starting training...")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs, device, output_dir)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")