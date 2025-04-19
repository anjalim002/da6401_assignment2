import argparse
import torch
import torch.nn as nn
import gc
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        filters = self.get_filter_list(config['filter_organisation'], config['total_filters'])
        activation = self.get_activation(config['activation'])
        self.use_bn = config['batch_normalisation'] == 'Yes'
        
        # Getting kernel sizes from config or using default pattern
        if 'kernel_sizes' in config:
            kernel_sizes = config['kernel_sizes']
        else:
            # By default all are set to 3
            kernel_sizes = [3, 3, 3, 3, 3]
            
        assert len(kernel_sizes) == 5, "Must provide exactly 5 kernel sizes for 5 conv layers"
        
        # 5 conv-activation-maxpool blocks
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=kernel_sizes[3], padding=kernel_sizes[3]//2)
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(filters[3], filters[4], kernel_size=kernel_sizes[4], padding=kernel_sizes[4]//2)
        self.bn5 = nn.BatchNorm2d(filters[4])
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters[4]*7*7, config['dense_layer_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.fc2 = nn.Linear(config['dense_layer_size'], 10)  # 10 output neurons
        self.activation = activation
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        if self.use_bn: x = self.bn3(x)
        x = self.activation(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        if self.use_bn: x = self.bn4(x)
        x = self.activation(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        if self.use_bn: x = self.bn5(x)
        x = self.activation(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        return self.fc2(x)
    
    def get_filter_list(self, mode, base):
        factor = {'same': 1, 'double': 2, 'half': 0.5}[mode]
        return [
            base,
            int(base*factor),
            int(base*(factor**2)),
            int(base*(factor**3)),
            int(base*(factor**4))
        ]
    
    def get_activation(self, name):
        return {
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU(),
            'SiLU': nn.SiLU(),
            'Mish': nn.Mish(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU()
        }[name]

def load_data(image_size, aug, batch_size, data_path="inaturalist_12K"):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
    
    base_tfms = [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        *base_tfms
    ]) if aug == 'Yes' else transforms.Compose(base_tfms)
    
    test_tfms = transforms.Compose(base_tfms)
    
    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_tfms)
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=test_tfms)
    
    targets = [s[1] for s in train_dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2000, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(train_dataset.dataset, val_idx)
    
    classes = train_dataset.dataset.classes
    
    return classes, DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

def train_model(config=None, use_wandb=True, save_best=False, save_path="best_model.pth"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if use_wandb:
        with wandb.init(config=config, project="DA6401_A2_PartA"):
            config = wandb.config

            # Update run name of wandb after initialization
            new_name = f"{config.optimizer}-{config.activation}-{config.filter_organisation}-{config.batch_size}"
            wandb.run.name = new_name

            return _train_core(config, use_wandb, save_best, save_path)
    else:
        return _train_core(config, use_wandb, save_best, save_path)

def _train_core(config, use_wandb=True, save_best=False, save_path="best_model.pth"):
    if isinstance(config, dict):
        aug = config.get('augmentation', 'Yes')
        batch_size = config.get('batch_size', 64)
        epochs = config.get('epochs', 10)
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.0001)
    else:
        # wandb.config
        aug = getattr(config, 'augmentation', 'Yes')
        batch_size = config.batch_size
        epochs = config.epochs
        optimizer_name = config.optimizer.lower()
        lr = config.learning_rate
        weight_decay = config.weight_decay

    # Load data
    classes, train_loader, val_loader, test_loader = load_data(
        image_size=224,
        aug=aug,
        batch_size=batch_size
    )

    model = CNN(config).to(device)

    # Setup optimizer
    optimizer_map = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'nadam': torch.optim.NAdam,
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop
    }

    optimizer_class = optimizer_map[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    # Tracking best model
    best_val_acc = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            del X, y, outputs, loss, preds

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        val_running_loss = 0.0

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                X, y = X.to(device), y.to(device)
                outputs = model(X)

                val_loss = criterion(outputs, y)
                val_running_loss += val_loss.item()
                val_correct += (outputs.argmax(1) == y).sum().item()
                val_total += y.size(0)
                del X, y, outputs, val_loss

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Log metrics
        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Saving best model
        if save_best and val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Load best model state
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'classes': classes
        }, save_path)
        print(f"Best model saved to {save_path} with validation accuracy: {best_val_acc:.4f}")

    return model, classes, test_loader

def test_model(model=None, test_loader=None, classes=None, model_path="best_model.pth"):
    if model is None:
        # Load saved model
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        classes = checkpoint['classes']

        model = CNN(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        _, _, _, test_loader = load_data(
            image_size=224,
            aug='No',  # No augmentation for testing
            batch_size=64
        )

    model.eval()
    test_correct, test_total = 0, 0
    all_preds = []
    all_targets = []
    all_images = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

            # Store predictions and targets for visualization
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_images.extend(X.cpu().numpy())

    test_accuracy = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy, all_preds, all_targets, all_images, classes

def visualize_predictions(all_images, all_preds, all_targets, classes, samples_per_class=2):
    # Convert images from normalized tensors back to displayable format
    def denormalize(x):
        # Assuming normalization was done with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = x * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
        return np.clip(x.transpose(1, 2, 0), 0, 1)

    # Create a figure with subplots - samples_per_class samples per class
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(samples_per_class*4, n_classes*3))

    for class_idx in range(n_classes):
        # Find indices where the true class is class_idx
        indices = np.where(np.array(all_targets) == class_idx)[0]

        if len(indices) < samples_per_class:
            print(f"Warning: Class {class_idx} ({classes[class_idx]}) has less than {samples_per_class} samples")
            samples = indices
        else:
            # Randomly select samples_per_class samples
            samples = np.random.choice(indices, samples_per_class, replace=False)

        for i, sample_idx in enumerate(samples):
            if samples_per_class == 1:
                ax = axes[class_idx]
            else:
                ax = axes[class_idx, i]

            # Display the image
            img = denormalize(all_images[sample_idx])
            ax.imshow(img)

            # Set title with true and predicted class
            true_class = classes[all_targets[sample_idx]]
            pred_class = classes[all_preds[sample_idx]]

            title = f"True: {true_class}\nPred: {pred_class}"
            if true_class == pred_class:
                ax.set_title(title, color='green')
            else:
                ax.set_title(title, color='red')

            ax.axis('off')

    plt.tight_layout()
    plt.savefig('class_predictions.png')
    plt.show()

def visualize_confusion_matrix(all_preds, all_targets, classes):
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Calculate class-wise metrics
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # Create a figure with two subplots - confusion matrix and metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 1]})

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14)

    # Normalize confusion matrix 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Convert NaN to 0

    # Create a table with metrics
    metrics_data = []
    for i, class_name in enumerate(classes):
        metrics_data.append([
            class_name,
            f"{cm[i, i]}/{sum(cm[i, :])}",
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{f1[i]:.4f}"
        ])

    # Add average metrics
    avg_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    metrics_data.append([
        "Average",
        f"{np.sum(np.diag(cm))}/{len(all_targets)}",
        f"{avg_precision:.4f}",
        f"{avg_recall:.4f}",
        f"{avg_f1:.4f}"
    ])

    # Create table for metrics
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(
        cellText=metrics_data,
        colLabels=["Class", "Correct/Total", "Precision", "Recall", "F1-score"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax2.set_title('Class-wise Performance Metrics', fontsize=14)

    plt.tight_layout()
    plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Normalized Confusion Matrix (Row-wise)', fontsize=14)
    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png', dpi=300)
    plt.show()

def run_wandb_sweep():
    """Run a wandb hyperparameter sweep"""
    sweep_config = {
        'name': 'optimizer-activation-filter_organisation-batchsize',
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'epochs': {'values': [5, 10, 15]},
            'batch_size': {'values': [32, 64, 128]},
            'learning_rate': {'min': 1e-5, 'max': 1e-3},
            'dropout': {'values': [0.1, 0.3]},
            'optimizer': {'values': ['adam', 'sgd', 'nadam', 'adamw', 'rmsprop']},
            'activation': {'values': ['ReLU', 'GELU', 'Mish', 'SiLU', 'ELU', 'SELU']},
            'augmentation': {'values': ['Yes', 'No']},
            'filter_organisation': {'values': ['same', 'double', 'half']},
            'dense_layer_size': {'values': [256, 512, 1024]},
            'total_filters': {'values': [32, 64]},
            'output_size': {'value': 10},  # Assuming 10 classes
            'batch_normalisation': {'values': ['Yes', 'No']},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
            'kernel_sizes': {
                'values': [
                    [3, 3, 3, 3, 3],  # All 3x3 kernels
                    [5, 5, 5, 5, 5],  # All 5x5 kernels
                    [3, 5, 3, 5, 3],  
                    [5, 3, 5, 3, 5],  
                    [3, 3, 5, 5, 7],  # Starting with small kernels, increasing to larger
                    [7, 5, 5, 3, 3],  # Starting with large kernels, decreasing to smaller
                    [3, 5, 7, 5, 3],  
                    [3, 5, 7, 11, 11],  
                    [11, 9, 7, 5, 3],  
                    [7, 5, 3, 5, 7],  
                ]
            }
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="DA6401_A2_PartA")
    wandb.agent(sweep_id, function=train_model, count=40)

def train_and_evaluate_specific_model():
    """Train and evaluate a model with specific configuration"""
    # Define specific configuration
    specific_config = {
        'epochs': 20, # for the highest val_accuracy i am increasing the epoch from 10 to 20 and taking same config to run the best hyperparameters
        'batch_size': 128,
        'learning_rate': 0.00029492,
        'dropout': 0.3,
        'optimizer': 'adam',
        'activation': 'SiLU',
        'augmentation': 'Yes',
        'filter_organisation': 'double',
        'dense_layer_size': 512,
        'total_filters': 32,
        'output_size': 10,
        'batch_normalisation': 'Yes',
        'weight_decay': 0.0014447,
        'kernel_sizes': [7,5,5,3,3]  
    }

    # Train the model with specific configuration
    model, classes, test_loader = train_model(
        config=specific_config,
        use_wandb=False,  
        save_best=True    
    )

    # Test the model and get metrics
    test_accuracy, all_preds, all_targets, all_images, classes = test_model(
        model=model,
        test_loader=test_loader,
        classes=classes
    )

    # Visualize results
    visualize_predictions(all_images, all_preds, all_targets, classes, samples_per_class=5)
    visualize_confusion_matrix(all_preds, all_targets, classes)

    return model, test_accuracy

def create_config_from_args(args):
    """Create a configuration dictionary from command line arguments"""
    # Convert filter_organisation format from filter list to mode
    if args.num_filters[1] == 2 * args.num_filters[0]:
        filter_org = 'double'
    elif args.num_filters[1] == int(0.5 * args.num_filters[0]):
        filter_org = 'half'
    else:
        filter_org = 'same'
    
    # Format activation function name to match CNN class expectation
    activation_map = {
        'relu': 'ReLU',
        'gelu': 'GELU',
        'silu': 'SiLU',
        'mish': 'Mish',
        'elu': 'ELU',
        'selu': 'SELU'
    }
    
    return {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'optimizer': args.optimizer,
        'activation': activation_map[args.activation],
        'augmentation': 'Yes' if args.augumentaion.lower() == 'yes' else 'No',
        'filter_organisation': filter_org,
        'dense_layer_size': args.dense_layer,
        'total_filters': args.num_filters[0],
        'output_size': 10,
        'batch_normalisation': 'Yes' if args.batch_norm.lower() == 'true' else 'No',
        'weight_decay': args.weight_decay,
        'kernel_sizes': args.filter_sizes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs23m065")
    parser.add_argument("--wandb_project", "-wp", help="Project name used to track experiments in Weights & Biases dashboard", default="Assignment 2")
    parser.add_argument("--epochs", "-e", help="Number of epochs to train neural network", type=int, default=10)
    parser.add_argument("--batch_size", "-b", help="Batch size used to train neural network", type=int, default=16)
    parser.add_argument("--optimizer", "-o", help="Optimizer used to train neural network", default="nadam", choices=['nadam', 'adam', 'sgd', 'adamw', 'rmsprop'])
    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", "-w_d", default=0.0001, type=float)
    parser.add_argument("--activation", "-a", choices=['relu', 'elu', 'selu', 'silu', 'gelu', 'mish'], default="relu")
    parser.add_argument("--num_filters", "-nf", nargs=5, type=int, default=[32, 32, 32, 32, 32])
    parser.add_argument("--filter_sizes", "-fs", nargs=5, type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument("--batch_norm", "-bn", default="true", choices=["true", "false"])
    parser.add_argument("--dense_layer", "-dl", default=128, type=int)
    parser.add_argument("--augumentaion", "-aug", default="yes", choices=["yes", "no"])
    parser.add_argument("--dropout", "-dp", default=0.2, type=float)
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K")
    parser.add_argument("--run_sweep", "-rs", action="store_true", help="Run WandB hyperparameter sweep")
    parser.add_argument("--save_model", "-sm", action="store_true", help="Save the best model")
    parser.add_argument("--model_path", "-mp", default="best_model.pth")
    
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    
    # Login to wandb
    wandb.login()
    
    if args.run_sweep:
        run_wandb_sweep()
    else:
        # Create config from args
        config = create_config_from_args(args)
        
        # Initialize wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config)
        
        # Train model
        model, classes, test_loader = train_model(
            config=config, 
            use_wandb=True, 
            save_best=args.save_model, 
            save_path=args.model_path
        )
        
        # Test and visualize
        test_accuracy, all_preds, all_targets, all_images, classes = test_model(
            model=model, 
            test_loader=test_loader, 
            classes=classes
        )
        
        # Visualize results
        visualize_predictions(all_images, all_preds, all_targets, classes)
        visualize_confusion_matrix(all_preds, all_targets, classes)
        
        wandb.finish()