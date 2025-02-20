import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from utils.DiceBCELoss import DiceBCELoss
from utils.dataset import BreastUltrasoundDataset, split_dataset
from models.unet_model import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def compute_metrics(preds, labels):
    # Ensure all tensors are on the same device
    device = preds.device
    labels = labels.to(device)

    # Flatten tensors
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    # Convert logits to probabilities
    preds_flat = torch.sigmoid(preds_flat)

    # Convert labels to binary (0 or 1) and ensure integer type
    labels_flat = (labels_flat > 0.5).int()

    # Threshold predictions to get binary masks
    binary_preds_flat = (preds_flat > 0.5).int()

    # Move tensors to the same device (redundant but ensures correctness)
    binary_preds_flat = binary_preds_flat.to(device)
    labels_flat = labels_flat.to(device)

    smooth = 1e-6

    # Compute metrics
    dice_score = 2.0 * torch.sum(binary_preds_flat * labels_flat) / (torch.sum(binary_preds_flat) + torch.sum(labels_flat) + smooth)
    iou_score = torch.sum(binary_preds_flat * labels_flat) / (torch.sum((binary_preds_flat + labels_flat) >= 1) + smooth)
    recall_score = torch.sum(binary_preds_flat * labels_flat) / (torch.sum(labels_flat) + smooth)
    precision_score = torch.sum(binary_preds_flat * labels_flat) / (torch.sum(binary_preds_flat) + smooth)
    global_accuracy = torch.sum(binary_preds_flat == labels_flat) / (binary_preds_flat.numel() + smooth)

    auc_roc = roc_auc_score(labels_flat.cpu().detach().numpy(), preds_flat.cpu().detach().numpy())
    if np.isnan(auc_roc): # Handle edge case when auc_roc is NaN
        auc_roc = 0.5

    metrics = {
        "dice_score": dice_score.item(),
        "iou_score": iou_score.item(),
        "recall": recall_score.item(),
        "precision": precision_score.item(),
        "global_accuracy": global_accuracy.item(),
        "auc_roc": auc_roc
    }

    return metrics

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, output_dir):
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
    best_dice_score = -float("inf")
    best_metrics = None

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

                # Compute metrics
                batch_metrics = compute_metrics(outputs, masks)
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
            if phase == "val" and epoch_metrics["dice_score"] > best_dice_score:
                best_dice_score = epoch_metrics["dice_score"]
                best_metrics = epoch_metrics
                print(f"\nBest model with Dice Score: {best_dice_score:.4f}")

        print()

    # Save final metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save best metrics
    with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=4)

    # Plot and save metrics graph
    plot_metrics(metrics, output_dir)

    return model

def plot_metrics(metrics, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(metrics["train_loss"], label="Train Loss")
    axes[0].plot(metrics["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(metrics["train_dice_score"], label="Train Dice Score")
    axes[1].plot(metrics["val_dice_score"], label="Validation Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_dice_graph.png"))
    plt.close()

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])  # IoU Score
    ax2 = fig.add_subplot(gs[0, 1])  # Recall
    ax3 = fig.add_subplot(gs[1, 0])  # Precision
    ax4 = fig.add_subplot(gs[1, 1])  # Global Accuracy

    ax1.plot(metrics["train_iou_score"], label="Train IoU Score")
    ax1.plot(metrics["val_iou_score"], label="Validation IoU Score")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("IoU Score")
    ax1.set_title("IoU Score")
    ax1.legend()
    ax1.grid()

    ax2.plot(metrics["train_recall"], label="Train Recall")
    ax2.plot(metrics["val_recall"], label="Validation Recall")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall")
    ax2.set_title("Recall")
    ax2.legend()
    ax2.grid()

    ax3.plot(metrics["train_precision"], label="Train Precision")
    ax3.plot(metrics["val_precision"], label="Validation Precision")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision")
    ax3.legend()
    ax3.grid()

    ax4.plot(metrics["train_global_accuracy"], label="Train Global Accuracy")
    ax4.plot(metrics["val_global_accuracy"], label="Validation Global Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Global Accuracy")
    ax4.set_title("Global Accuracy")
    ax4.legend()
    ax4.grid()

    # Last plot (AUC ROC) will span two columns in the last row
    ax5 = fig.add_subplot(gs[2, :])  # Spanning both columns
    ax5.plot(metrics["train_auc_roc"], label="Train AUC ROC")
    ax5.plot(metrics["val_auc_roc"], label="Validation AUC ROC")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("AUC ROC")
    ax5.set_title("AUC ROC")
    ax5.legend()
    ax5.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "other_metrics_graph.png"))
    plt.close()



def log_training_details(output_dir, params):
    with open(os.path.join(output_dir, "training_log.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    # Paths and Hyperparameters
    root_dir = "./dataset"
    output_dir = "./training_outputs"  # Output directory for saving models and metrics
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-6
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
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Log training details
    training_params = {
        "Dataset Path": root_dir,
        "Output Directory": output_dir,
        "Number of Epochs": num_epochs,
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "Weight Decay": weight_decay,
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