# ESE : Exemplar Storage & Extraction frameworks for Continual Learning

## Introduction

Continual learning, also known as lifelong learning, is a paradigm in machine learning where models are trained to learn continuously from a stream of data. Unlike traditional learning methods that assume all data is available at once, continual learning models must adapt to new information while retaining previously learned knowledge. This approach is crucial for developing intelligent systems that can evolve and improve over time, similar to human learning.

In continual learning, one of the main challenges is overcoming catastrophic forgetting, where the model forgets previously learned information upon learning new tasks. To address this, various strategies such as exemplar storage and extraction are employed. These strategies help in maintaining a balance between learning new information and retaining old knowledge.

For those interested in exploring this field further, the following papers provide a comprehensive overview and insights into continual learning:
- [Continual Learning: A Comprehensive Review](https://arxiv.org/abs/1802.07569)
- [Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)
- [Gdumb: A Simple Approach that Questions Our Progress in Continual Learning](https://arxiv.org/abs/1910.07104)
- [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)
- [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486)

## Overview

A framework for continual learning using exemplar storage and extraction.  
`2024.10.26` : first public release

## Features

- Supporting continual learning methods `er`, `er_ace`, `der`, `der++`
- Supporting datasets `mnist`, `cifar10`, `cifar100`
- Supporting experimental settings `cil`:class incremental learning, `dil`:domain incremental learning
- Supporting exemplar(buffer) extraction and storage strategies 
  - exemplar extraction: `random`, `mir`
  - exemplar storage: `random`, `gss`

## Installation with Docker
### Dockerfile
```
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get upgrade -y
RUN apt-get install pip git -y
RUN apt-get install pip --upgrade
RUN apt-get install tmux -y
RUN pip install tqdm gpustat scikit-learn wandb matplotlib  seaborn
```

### Build Docker Image & Run Container
```
docker build -t ese .
docker run -it --gpus all --name ese_container ese
```

## How to Run

### RUN Example
```
python main.py --dataset mnist --cl_type cil --model er --buffer_extraction random --buffer_storage random
```

### Configuration Options

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


## Contribution

Hyeonchang Chu in [AI_LAB](http://air.cau.ac.kr/)

## License

MIT License

## Contact

email: cngusckd@gmail.com