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


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, output_dir):
    """
    Train the model with progress tracking for each epoch and additional metrics logging.
    """
    metrics = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

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

                # Optionally update progress bar with current loss
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / total_samples
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            # Log metrics
            metrics[f"{phase}_loss"].append(epoch_loss)

            # Save best model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"\nBest model saved with loss: {best_loss:.4f}")

        print()

    # Save final metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot and save metrics graph
    plot_metrics(metrics, output_dir)

    return model

def plot_metrics(metrics, output_dir):
    """
    Plot and save training and validation loss graphs.
    """
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

def log_training_details(output_dir, params):
    """
    Save training metadata to a log file.
    """
    with open(os.path.join(output_dir, "training_log.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    # Paths and Hyperparameters
    root_dir = "./dataset"  # Path to your dataset
    output_dir = "./training_outputs"  # Output directory for saving models and metrics
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = 25
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    splits = split_dataset(root_dir)
    transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])

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
    model = UNet(in_channels=3, out_channels=1).to(device)  # Assuming RGB images
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
