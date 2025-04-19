# iNaturalist Image Classification Project

This repository contains a two-part implementation for image classification on the iNaturalist dataset:
- **Part A**: Custom CNN model implementation with hyperparameter tuning
- **Part B**: Transfer learning with ResNet50 using various fine-tuning strategies

### Dataset

The script can automatically download the iNaturalist_12K dataset or you can provide your own. The dataset should be organized as follows:

```
inaturalist_12K/
├── train/
│   ├── Amphibia/
│   ├── Animalia/
│   ├── Arachnida/
│   ├── Aves/
│   ├── Fungi/
│   ├── Insecta/
│   ├── Mammalia/
│   ├── Mollusca/
│   ├── Plantae/
│   └── Reptilia/
└── val/
    ├── Amphibia/
    ├── Animalia/
    ├── Arachnida/
    ├── Aves/
    ├── Fungi/
    ├── Insecta/
    ├── Mammalia/
    ├── Mollusca/
    ├── Plantae/
    └── Reptilia/
```

## Part A: Custom CNN Implementation

### Features

- Custom CNN architecture with configurable parameters
- Hyperparameter tuning with Weights & Biases sweeps
- Multiple activation functions support (ReLU, GELU, SiLU, Mish, ELU, SELU)
- Different filter organization strategies (same, double, half)
- Configurable kernel sizes
- Batch normalization
- Data augmentation options
- Detailed visualization of results with confusion matrices and class predictions

### Requirements

```
torch
torchvision
numpy
matplotlib
seaborn
sklearn
wandb
tqdm
```

### Usage

#### Basic Training

```bash
python partA.py --wandb_entity your_username --wandb_project your_project_name
```

#### Training with Custom Parameters

```bash
python partA.py --epochs 15 --batch_size 64 --optimizer adam --learning_rate 0.0003 --dropout 0.3 --activation silu --batch_norm true --augumentaion yes
```

#### Run Hyperparameter Sweep

```bash
python partA.py --run_sweep
```

#### Save Best Model During Training

```bash
python partA.py --save_model --model_path "my_best_model.pth"
```

### Command Line Arguments

```
--wandb_entity, -we      Wandb Entity for tracking experiments
--wandb_project, -wp     Project name in Weights & Biases dashboard
--epochs, -e             Number of training epochs (default: 10)
--batch_size, -b         Batch size (default: 16)
--optimizer, -o          Optimizer (choices: nadam, adam, sgd, adamw, rmsprop; default: nadam)
--learning_rate, -lr     Learning rate (default: 0.001)
--weight_decay, -w_d     Weight decay (default: 0.0001)
--activation, -a         Activation function (choices: relu, elu, selu, silu, gelu, mish; default: relu)
--num_filters, -nf       Number of filters for each of the 5 conv layers (default: [32, 32, 32, 32, 32])
--filter_sizes, -fs      Kernel sizes for each of the 5 conv layers (default: [3, 3, 3, 3, 3])
--batch_norm, -bn        Use batch normalization (choices: true, false; default: true)
--dense_layer, -dl       Size of dense layer (default: 128)
--augumentaion, -aug     Use data augmentation (choices: yes, no; default: yes)
--dropout, -dp           Dropout rate (default: 0.2)
--base_dir, -br          Base directory for dataset (default: inaturalist_12K)
--run_sweep, -rs         Run WandB hyperparameter sweep
--save_model, -sm        Save the best model
--model_path, -mp        Path to save the best model (default: best_model.pth)
```

### Output

The script will:
1. Load and prepare the iNaturalist dataset
2. Configure and train the CNN model
3. Evaluate the model on test data
4. Generate visualizations:
   - Sample predictions
   - Confusion matrix
   - Class-wise performance metrics

## Part B: Transfer Learning with ResNet50

### Features

- Fine-tunes ResNet50 on the iNaturalist dataset (10 classes)
- Supports multiple fine-tuning strategies:
  - Freeze all layers except last (classification head)
  - Freeze early layers
  - Progressive unfreezing
  - Differential learning rates
  - Full fine-tuning
- Tracks experiments with Weights & Biases
- Automatically downloads dataset if needed
- Configurable hyperparameters

### Requirements

```
torch
torchvision
numpy
matplotlib
tqdm
sklearn
wandb
```

### Usage

Basic usage:

```bash
python partB.py --wandb_entity your_username --wandb_project your_project_name
```

Download dataset and run:

```bash
python partB.py --wandb_entity your_username --wandb_project your_project_name --download_dataset
```

Specify W&B API key:

```bash
python partB.py --wandb_entity your_username --wandb_project your_project_name --wandb_key your_api_key
```

### Fine-Tuning Strategies

Choose a fine-tuning strategy:

```bash
python partB.py --strategy freeze_early_layers
```

Available strategies:
- `freeze_all_except_last`: Only trains the classification head
- `freeze_early_layers`: Freezes early layers(20 in my code), trains later layers
- `progressive_unfreeze`: Gradually unfreezes layers during training
- `differential_learning_rates`: Uses different learning rates for different layers
- `full`: Fine-tunes the entire model

### All Command Line Arguments

```
--wandb_entity, -we      Wandb Entity used to track experiments
--wandb_project, -wp     Project name used to track experiments
--epochs, -e             Number of epochs to train neural network (default: 10)
--batch_size, -b         Batch size for training (default: 32)
--optimizer, -o          Optimizer to use (choices: nadam, adam, rmsprop; default: adam)
--learning_rate, -lr     Learning rate for optimizer (default: 0.001)
--weight_decay, -w_d     Weight decay for optimizer (default: 0.0001)
--base_dir, -br          Base directory for dataset (default: inaturalist_12K)
--strategy, -s           Fine-tuning strategy to use (default: freeze_all_except_last)
--download_dataset, -dd  Download the dataset
--wandb_key, -wk         Wandb API key
```

### Example Commands

Basic training with default parameters:
```bash
python partB.py --wandb_entity username --wandb_project iNaturalist
```

Training with custom parameters:
```bash
python partB.py --wandb_entity username --wandb_project iNaturalist --epochs 20 --batch_size 64 --optimizer adam --learning_rate 0.0005 --strategy progressive_unfreeze
```

Download dataset and train with RMSprop:
```bash
python partB.py --wandb_entity username --wandb_project iNaturalist --download_dataset --optimizer rmsprop --learning_rate 0.0002
```

Github Link:
https://github.com/anjalim002/da6401_assignment2/tree/master 
