# CLN : Continual Learning framework for Novice

## Introduction

Continual learning, also known as lifelong learning, is a paradigm in machine learning where models are trained to learn continuously from a stream of data. Unlike traditional learning methods that assume all data is available at once, continual learning models must adapt to new information while retaining previously learned knowledge. This approach is crucial for developing intelligent systems that can evolve and improve over time, similar to human learning.

In continual learning, one of the main challenges is overcoming catastrophic forgetting, where the model forgets previously learned information upon learning new tasks. To address this, various strategies such as exemplar storage and extraction are employed. These strategies help in maintaining a balance between learning new information and retaining old knowledge.

### Continual Learning Mechanism

The continual learning process can be visualized as follows:

1. **Task Stream**: The model receives a sequence of tasks, each with its own dataset.
   ```
   Task 0 -> Task 1 -> Task 2 -> ... -> Task N
   ```

2. **Learning and Evaluation**: For each task, the model learns from the current task's data and is evaluated on both the current and previous tasks to ensure knowledge retention.
   ```
   [Learn Task 0] -> [Evaluate on Task 0]
   [Learn Task 1] -> [Evaluate on Task 0 ~ 1]
   [Learn Task 2] -> [Evaluate on Task 0 ~ 2]
   ```

For those interested in exploring this field further, the following papers provide a comprehensive overview and insights into continual learning:
- [Continual Learning: A Comprehensive Review](https://arxiv.org/abs/1802.07569)
- [Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)
- [Gdumb: A Simple Approach that Questions Our Progress in Continual Learning](https://arxiv.org/abs/1910.07104)
- [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)
- [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486)

## Timeline for versions

`v1.1` : first public release (`2024.10.30` )

## Features

- Supporting continual learning methods `er`, `er_ace`, `der`, `der++`
- Supporting datasets `mnist`, `cifar10`, `cifar100`, `custom_dataset`
- Supporting experimental settings `cil`:class incremental learning, `dil`:domain incremental learning
- Supporting backbone models `resnet18`, `resnet34`, `resnet50`
- Supporting exemplar(buffer) extraction and storage strategies 
  - exemplar extraction: `random`, `mir` (methods used to decide which data to extract from the buffer during replay)
  - exemplar storage: `random`, `gss` (methods used to decide which data to replace in the buffer storage)

## Installation with Docker
### Dockerfile
```
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get upgrade -y
RUN apt-get install pip git -y
RUN apt-get install pip --upgrade
RUN apt-get install tmux wget unzip tree -y
RUN pip install tqdm gpustat scikit-learn wandb matplotlib seaborn

RUN git clone https://github.com/cngusckd/ESE-framework.git
```

### Build Docker Image & Run Container
```
docker build -t ese .
docker run -it --gpus all --name ese_container ese
```

## How to Run ESE framework

### RUN Example
```
python main.py --seed 42--dataset mnist --cl_type cil --model er --buffer_extraction random --buffer_storage random
```
This command initiates the overall learning sequence for continual learning, where validation is performed at the end of each task (continual learning consists of a sequence of multiple tasks). At the end of each task, the model evaluates its performance using validation data, allowing us to check how well the model retains previously learned information while learning new information. This process is crucial for preventing catastrophic forgetting and ensuring the continuous improvement of the model's performance.  

If you want to conduct experiments under the same conditions but with different random seeds, you can modify the `--seed` option to different numbers. This allows you to test the robustness and variability of your model's performance across different random initializations. By running multiple experiments with different seeds, you can gain insights into the stability and generalizability of your model, which is crucial for AI research. This practice helps in identifying whether the observed performance is consistent or if it varies significantly due to the randomness in the training process.

When you run the framework with the `--wandb` option, several performance metrics are logged to Weights & Biases ([wandb](#integrating-weights--biases-wandb)). These metrics help in understanding the model's performance and behavior during training and evaluation:

- **Accuracy (val_acc)**: This metric indicates the proportion of correctly classified instances out of the total instances. It provides a general sense of how well the model is performing on the validation set.

- **Validation Loss (val_loss)**: This metric measures the error of the model on the validation set. Lower values indicate better performance, as the model's predictions are closer to the actual values.

- **AUROC (Area Under the Receiver Operating Characteristic Curve)**: This metric evaluates the model's ability to distinguish between classes. A higher AUROC value indicates better performance in terms of classification capability.

- **Confusion Matrix**: This is a table used to describe the performance of a classification model. It shows the true vs. predicted classifications, helping to identify specific areas where the model may be making errors.

- **Buffer Distribution**: This metric shows the distribution of classes in the model's buffer. It helps in understanding how well the buffer is maintaining a balanced representation of different classes.

- **ROC Curve**: The Receiver Operating Characteristic curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system. It is used to visualize the trade-off between the true positive rate and false positive rate.

These metrics are visualized in the wandb dashboard, allowing you to track and analyze the model's performance over time. You can customize the wandb dashboard to focus on specific metrics or compare different runs.

### Configuration Options

- `--seed`: Seed for experiment (default: 42)
- `--wandb`: Use wandb for experiment
- `--dataset`: Dataset to use ('mnist', 'cifar10', 'cifar100')
- `--image_shape`: Shape of the dataset images (default: (32, 32))
- `--cl_type`: Type of continual learning ('cil', 'dil')
- `--nclasses`: Number of classes in the dataset (default: 10)
- `--num_increments`: Number of tasks in continual learning (default: 5)
- `--device`: Device for deep learning ('cpu', 'cuda')
- `--epoch`: Number of epochs per task (default: 10)
- `--batch_size`: Batch size for current data stream (default: 64)
- `--buffer_memory_size`: Size of the buffer (default: 500)
- `--model`: Continual learning method to use ('er', 'er_ace', 'der', 'der++')
- `--buffer_extraction`: Buffer extraction strategy ('random', 'mir')
- `--buffer_extraction_size`: Size of the buffer extraction batch (default: 64)
- `--buffer_storage`: Buffer storage strategy ('random', 'gss')
- `--buffer_storage_size`: Size of the buffer storage update (default: 64)
- `--optim`: Optimizer for learning ('sgd')
- `--lr`: Learning rate for the optimizer (default: 1e-3)
- `--momentum`: Momentum for the optimizer (default: 0.9)
- `--alpha`: Hyperparameter for knowledge distillation in DER, DER++ (default: 0.1)
- `--beta`: Hyperparameter for knowledge distillation in DER++ (default: 0.5)

### Key Components

- `model/trainer.py`: Manages the training process.
- `model/continual_model.py`: Defines the base continual learning model class.
- `model/er,er_ace,der`: Implements each continual learning method.
- `model/buffer.py`: Implements buffer for exemplar storage and extraction.
- `data/dataloader.py`: Implements data loader for incremental learning.

## Integrating Weights & Biases (wandb)

`wandb` is a tool for experiment tracking, model optimization, and dataset versioning. Follow these steps to integrate `wandb` with your experiments:

### Step 1: Set Up `wandb` from Bash

1. **Create a `wandb` Account**:
   - Visit [wandb.ai](https://wandb.ai) and sign up for a free account.

2. **Install `wandb`**:
   - If not already installed, run the following command in your terminal:
     ```bash
     pip install wandb
     ```

3. **Login to `wandb`**:
   - In your terminal, execute:
     ```bash
     wandb login
     ```
   - This will prompt you to enter an API key. You can find your API key in your `wandb` account settings under the "API Keys" section.


### Step 2: Run Your Experiment

Run your script with the `--wandb` flag to enable logging:

```bash
python main.py --dataset mnist --cl_type cil --model er --buffer_extraction random --buffer_storage random --wandb
```

### Step 3: Monitor Your Experiments

1. **Access the `wandb` Dashboard**:
   - Log in to your `wandb` account and navigate to the dashboard. You can do this by visiting [wandb.ai](https://wandb.ai) and clicking on your project.

2. **Explore Your Runs**:
   - In the dashboard, you will see a list of your runs. Click on a run to view detailed metrics, logs, and visualizations.

3. **Visualize Metrics**:
   - Use the interactive plots to visualize metrics such as loss, accuracy, AUROC, and more. You can customize these plots to compare different runs or focus on specific metrics.

4. **Analyze Images and Graphs**:
   - View logged images, such as confusion matrices and ROC curves, directly in the dashboard. This helps in understanding model performance visually.

## Using IncrementalCustomDataloader

The framework supports Incremental Learning for custom datasets like Tiny ImageNet using the `IncrementalCustomDataloader` class. This allows you to load datasets incrementally by class or domain.

### Preparing the Tiny ImageNet Dataset for IncrementalCustomDataloader

First, use the `make_tiny_image.sh` script to download and prepare the Tiny ImageNet dataset.
```
bash make_tiny_image.sh
```

### Directory Structure

After running the script, the Tiny ImageNet dataset will be organized as follows:

```
data/
└── tiny_imagenet/
└── tiny-imagenet-200/
├── train/
│ ├── n01443537/
│ │ ├── images/
│ │ └── ...
│ ├── n01629819/
│ │ ├── images/
│ │ └── ...
│ └── ...
└── val/
├── n01443537/
│ ├── images/
│ └── ...
├── n01629819/
│ ├── images/
│ └── ...
└── ...
```

### Example Configuration for Tiny ImageNet

When using the Tiny ImageNet dataset, you need to specify the `image_shape` and `nclasses` options. Below is an example of how to use Tiny ImageNet:

```
python main.py --dataset custom_dataset --image_shape 64 64 --nclasses 200
```

### Details for IncrementalCustomDataloader

The structure of the custom dataset is organized such that each folder represents a class, and the images within each folder correspond to that specific class. This means that the dataset is divided into multiple folders, where each folder contains images belonging to a particular class.

To use IncrementalCustomDataloader, you need to modify the `root` argument in `model/trainer.py` to match the location of your dataset.

For example, if you have a dataset with classes such as 'cat', 'dog', and 'bird', the directory structure would look like this:

```
data/
└── custom_dataset/
    ├── train/
    │   ├── cat/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── dog/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── bird/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    └── val/
        ├── cat/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── dog/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── bird/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
```

This structure allows the framework to easily load and process images for each class during training and validation.

## Testing Framework with Shell Scripts

To automate the testing of our continual learning framework with different configurations, we provide two shell scripts: `test.sh` and `multi_process_test.sh`.

### `test.sh`
This script tests the framework by iterating over all possible combinations of configuration options sequentially. It generates combinations of dataset, continual learning type, model, buffer extraction strategy, and buffer storage strategy, and then runs the `main.py` script with each combination.

```bash
# Example usage
bash test.sh
```

### `multi_process_test.sh`
This script performs the same task as `test.sh`, but executes multiple combinations in parallel to speed up the testing process. It utilizes `xargs` to run multiple processes concurrently.

```bash
# Example usage
bash multi_process_test.sh
```

- **Key Features**:
  - Uses Python to generate all possible combinations of configuration options.
  - Supports parallel execution with up to 5 processes.


## Contribution

Hyeonchang Chu in [AI_LAB](http://air.cau.ac.kr/)   
Thanks to Hyejeong Im, Seungwoo Song for feedback :)
## License

MIT License

## Contact

email: cngusckd@gmail.com
