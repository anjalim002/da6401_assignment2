import os
import argparse
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Global variables
dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define class labels for iNaturalist dataset
classesList = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

# Model strategy for fine-tuning
class ModelStrategy:
    def __init__(self, strategy_name='full'):
        self.strategy_name = strategy_name

    def setup_model(self):
        # Load pretrained ResNet50
        model = models.resnet50(pretrained=True)

        # Strategy 1: Freeze all layers and only train the new classification head
        if self.strategy_name == 'freeze_all_except_last':
            # Freeze all parameters in the model
            for param in model.parameters():
                param.requires_grad = False

            # Replace the final fully connected layer for 10 classes
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 10)

        # Strategy 2: Freeze only a certain number of early layers
        # elif self.strategy_name == 'freeze_early_layers':
        #     # Freeze the first 6 layers (conv1, bn1, and 4 layer1 blocks)
        #     layers_to_freeze = list(model.children())[:6]
        #     for layer in layers_to_freeze:
        #         for param in layer.parameters():
        #             param.requires_grad = False

        #     # Replace final FC layer
        #     num_features = model.fc.in_features
        #     model.fc = nn.Linear(num_features, 10)
        elif self.strategy_name == 'freeze_early_layers':
            # Get all layers in the model
            all_layers = list(model.children())

            # Freeze the first 20 layers instead of just 6
            # ResNet50 has a more complex structure with nested layers
            # To freeze approximately 20 layers, we need to go deeper into the model structure

            # First, freeze the initial layers (conv1, bn1, relu, maxpool)
            for layer in all_layers[:4]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Then freeze layers in layer1, layer2, layer3 (these are sequential blocks that contain multiple layers each)
            for block_idx in range(4, 7):  # This targets layer1, layer2, layer3
                block = all_layers[block_idx]
                for sublayer in block:  # Each block contains multiple bottleneck/basic blocks
                    for param in sublayer.parameters():
                        param.requires_grad = False

            # Replace final FC layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 10)
            
        # Strategy 3: Progressive unfreezing (initially freeze all, then gradually unfreeze)
        elif self.strategy_name == 'progressive_unfreeze':
            # Initially freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Replace final FC layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 10)

            # Note: We'll unfreeze progressively during training
            self.model_layers = list(model.children())

        # Strategy 4: Use lower learning rate for pretrained layers and higher for new layers
        elif self.strategy_name == 'differential_learning_rates':
            # Replace final FC layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 10)

            # All parameters will be trainable but with different learning rates
            # We'll handle this in the optimizer setup

        # Strategy 5: Full fine-tuning (train all parameters, no freezing)
        else:  
            # Replace final FC layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 10)
            # No freezing - all parameters will be trained

        return model.to(device)

    def get_param_groups(self, model, base_lr=0.001):
        """For differential learning rates strategy"""
        if self.strategy_name == 'differential_learning_rates':
            # Group parameters: pretrained layers get lower lr, new layers get higher lr
            pretrained_params = []
            new_params = []

            for name, param in model.named_parameters():
                if 'fc' in name:  # New layer parameters
                    new_params.append(param)
                else:  # Pretrained layer parameters
                    pretrained_params.append(param)

            param_groups = [
                {'params': pretrained_params, 'lr': base_lr * 0.1},  # Lower LR for pretrained
                {'params': new_params, 'lr': base_lr}                # Higher LR for new
            ]
            return param_groups
        else:
            return model.parameters()

    def progressive_unfreeze_step(self, model, current_epoch, total_epochs):
        """For progressive unfreezing strategy"""
        if self.strategy_name == 'progressive_unfreeze' and current_epoch > 0:
            # Calculate how many layers to unfreeze based on current epoch
            # Unfreeze from back to front (deep to shallow)
            layers_to_unfreeze = int((current_epoch / total_epochs) * len(self.model_layers))

            # Start unfreezing from the end (excluding the FC layer which is always trainable)
            for i in range(len(self.model_layers) - 1, len(self.model_layers) - 1 - layers_to_unfreeze, -1):
                if i >= 0:  
                    for param in self.model_layers[i].parameters():
                        param.requires_grad = True

# Training and evaluation functions
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, strategy=None, wandb_project="iNaturalist-Fine-Tuning"):
    """Train the model with the given dataloaders, criterion, and optimizer"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0

    # Initialize wandb for tracking experiments if not already initialized
    if wandb.run is None:
        wandb.init(project=wandb_project, name=f"ResNet50-{strategy.strategy_name}")
        wandb.config.update({
            "model": "ResNet50",
            "strategy": strategy.strategy_name,
            "epochs": num_epochs
        })

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']

                # For progressive unfreezing, update which layers are frozen
                if strategy.strategy_name == 'progressive_unfreeze':
                    strategy.progressive_unfreeze_step(model, epoch, num_epochs)

                    # Log how many parameters are now trainable
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"Epoch {epoch+1}: Trainable parameters: {trainable_params:,}")
                    wandb.log({"trainable_params": trainable_params}, step=epoch)
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass - only track history in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to wandb
            wandb.log({f"{phase}_loss": epoch_loss,
                       f"{phase}_accuracy": epoch_acc}, step=epoch)

            # Track metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

                # Save the best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), f'best_model_{strategy.strategy_name}.pth')
                    print(f'New best validation accuracy: {best_acc:.4f}')

    print(f'Best validation Acc: {best_acc:.4f}')

    # Plot training history
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'training_history_{strategy.strategy_name}.png')
        plt.close()
    except Exception as e:
        print(f"Could not create plots: {e}")
        print("Continuing without plotting...")

    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, dataloader):
    """Evaluate the model on the given dataloader"""
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / len(dataloader.dataset)
    print(f'Test Accuracy: {acc:.4f}')

    return acc.item()

# Function to prepare the data
def prepare_data(data_dir, batch_size=32):
    # ResNet50 expects 224x224 images, so we resize all input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ImageNet dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load training data
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    
    # Create stratified train/validation split from training data
    train_indices, val_indices = [], []
    
    for class_idx in range(len(train_dataset.classes)):
        class_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == class_idx]
        class_train_idx, class_val_idx = train_test_split(
            class_indices,
            test_size=0.2,  
            random_state=42
        )
        
        train_indices.extend(class_train_idx)
        val_indices.extend(class_val_idx)
    
    # Create data subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Test data - using the val folder strictly for testing only
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training set size: {len(train_subset)} images")
    print(f"Validation set size: {len(val_subset)} images")
    print(f"Test set size: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader

# Function to print class distribution
def print_class_distribution(dataloader, title):
    class_counts = {i: 0 for i in range(len(classesList))}
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1
    
    print(f"\n{title} class distribution:")
    for i, count in class_counts.items():
        print(f"{classesList[i]}: {count} images")

# Function to create a model from scratch
def create_scratch_model():
    # Create a ResNet50 without pretrained weights
    model = models.resnet50(pretrained=False)
    # Modify final layer for 10 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model.to(device)

# Main configuration and training function
def config_and_train(base_dir="inaturalist_12K", learning_rate=0.001, weight_decay=0.0001, 
                   epochs=10, batchSize=32, optimiser_fn="adam", strategy_name="freeze_early_layers", 
                   wandb_project="iNaturalist-Fine-Tuning"):
    
    print(f"Using device: {device}")
    print(f"Strategy: {strategy_name}")
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_data(base_dir, batch_size=batchSize)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Print class distribution
    print_class_distribution(train_loader, "Training")
    print_class_distribution(val_loader, "Validation")
    
    # Set up model strategy
    model_strategy = ModelStrategy(strategy_name)
    model = model_strategy.setup_model()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Fixed parameters: {total_params - trainable_params:,}")
    print(f"Strategy used: {strategy_name}")
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Get parameter groups (for differential learning rates if needed)
    param_groups = model_strategy.get_param_groups(model, base_lr=learning_rate)
    
    # Choose optimizer based on input
    if optimiser_fn.lower() == "adam":
        if strategy_name == 'differential_learning_rates':
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(param_groups, lr=learning_rate, weight_decay=weight_decay)
    elif optimiser_fn.lower() == "nadam":
        if strategy_name == 'differential_learning_rates':
            optimizer = optim.NAdam(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.NAdam(param_groups, lr=learning_rate, weight_decay=weight_decay)
    elif optimiser_fn.lower() == "rmsprop":
        if strategy_name == 'differential_learning_rates':
            optimizer = optim.RMSprop(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.RMSprop(param_groups, lr=learning_rate, weight_decay=weight_decay)
    else:
        # Default to Adam
        if strategy_name == 'differential_learning_rates':
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(param_groups, lr=learning_rate, weight_decay=weight_decay)
    
    # Prepare dataloaders dictionary
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=epochs,
        strategy=model_strategy,
        wandb_project=wandb_project
    )
    
    # Evaluate the best model on test data
    print("\nEvaluating best model on test data...")
    model.load_state_dict(torch.load(f'best_model_{strategy_name}.pth'))
    test_acc = evaluate_model(model, test_loader)
    print(f"Final test accuracy: {test_acc:.4f}")
    
    # Compare with model trained from scratch if requested
    compare_with_scratch = False
    if compare_with_scratch:
        print("\n=== Fine-Tuning vs Training from Scratch Analysis ===")
        
        # Initialize a model from scratch
        scratch_model = create_scratch_model()
        
        # Set up optimizer for training from scratch
        scratch_optimizer = optim.Adam(scratch_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Train the scratch model
        print("\nTraining model from scratch...")
        wandb.init(project=wandb_project, name="ResNet50-from-scratch")
        wandb.config.update({
            "model": "ResNet50",
            "strategy": "from-scratch",
            "epochs": epochs
        })
        
        scratch_model, s_train_losses, s_val_losses, s_train_accs, s_val_accs = train_model(
            scratch_model,
            dataloaders,
            criterion,
            scratch_optimizer,
            num_epochs=epochs,
            strategy=ModelStrategy("from-scratch"),
            wandb_project=wandb_project
        )
        
        # Evaluate the best scratch model
        scratch_model.load_state_dict(torch.load('best_model_from-scratch.pth'))
        scratch_test_acc = evaluate_model(scratch_model, test_loader)
        
        # Compare results
        print("\n=== Comparison of Results ===")
        print(f"Fine-tuned model ({strategy_name}) test accuracy: {test_acc:.4f}")
        print(f"Model trained from scratch test accuracy: {scratch_test_acc:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(val_accs, 'b-', label=f'Fine-tuned ({strategy_name})')
        plt.plot(s_val_accs, 'r-', label='From scratch')
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('comparison_plot.png')
        plt.close()
        
        # Insights and analysis
        print("\n=== Insights from Fine-Tuning vs Training from Scratch ===")
        print("1. Learning Speed: Fine-tuning typically converges faster because the model")
        print("   already has learned useful representations from ImageNet.")
        print("2. Final Performance: The pre-trained model usually achieves better final")
        print("   accuracy as it leverages knowledge from a much larger dataset.")
        print("3. Data Efficiency: Fine-tuning requires less training data to achieve good")
        print("   performance, making it ideal for smaller datasets.")
        print("4. Training Time: Fine-tuning is computationally more efficient and requires")
        print("   fewer epochs to reach good performance.")
        print("5. Transfer Learning Power: The knowledge transfer from ImageNet to iNaturalist")
        print("   is particularly strong because of the shared domain (both contain natural images).")
    
    return test_acc

def download_dataset(download_url="https://storage.googleapis.com/wandb_datasets/nature_12K.zip"):
    import subprocess
    
    print("Downloading dataset...")
    subprocess.run(["wget", download_url, "-O", "nature_12K.zip"])
    
    print("Extracting dataset...")
    subprocess.run(["unzip", "-q", "nature_12K.zip"])
    
    print("Dataset ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning ResNet50 on iNaturalist dataset")
    parser.add_argument("--wandb_entity", "-we", help="Wandb Entity used to track experiments", default="cs23m065")
    parser.add_argument("--wandb_project", "-wp", help="Project name used to track experiments", default="Assignment 2")
    parser.add_argument("--epochs", "-e", help="Number of epochs to train neural network", type=int, default=10)
    parser.add_argument("--batch_size", "-b", help="Batch size used to train neural network", type=int, default=32)
    parser.add_argument("--optimizer", "-o", help="Optimizer to use for training", default="adam", 
                        choices=['nadam', 'adam', 'rmsprop'])
    parser.add_argument("--learning_rate", "-lr", help="Learning rate for optimizer", default=0.001, type=float)
    parser.add_argument("--weight_decay", "-w_d", help="Weight decay for optimizer", default=0.0001, type=float)
    parser.add_argument("--base_dir", "-br", help="Base directory for dataset", default="inaturalist_12K")
    parser.add_argument("--strategy", "-s", help="Fine-tuning strategy to use", 
                        default="freeze_all_except_last", 
                        choices=['freeze_all_except_last', 'freeze_early_layers', 
                                'progressive_unfreeze', 'differential_learning_rates', 'full'])
    parser.add_argument("--download_dataset", "-dd", help="Download the dataset", action="store_true")
    parser.add_argument("--wandb_key", "-wk", help="Wandb API key", default=None)
    
    args = parser.parse_args()
    
    # Download dataset if requested
    if args.download_dataset:
        download_dataset()
    
    # Login to wandb if key is provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    else:
        wandb.login()
    
    # Start a new wandb run
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    # Run training
    config_and_train(
        base_dir=args.base_dir, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        epochs=args.epochs, 
        batchSize=args.batch_size, 
        optimiser_fn=args.optimizer,
        strategy_name=args.strategy,
        wandb_project=args.wandb_project
    )
    
    # Finish wandb run
    wandb.finish()