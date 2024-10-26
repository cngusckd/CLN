# ESE : Exemplar Storage & Extraction frameworks for Continual Learning

## Overview

A framework for continual learning using exemplar storage and extraction.
`2024.10.26` : first public release

## Features

- Supporting continual learning methods `er`, `er_ace`, `der`, `der++`
- Supporting datasets `mnist`, `cifar10`, `cifar100`
- Supporting experimental settings `cil`:class incremental learning, `dil`:domain incremental learning
- Various exemplar(buffer) extraction and storage strategies 
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
RUN pip install tqdm gpustat scikit-learn
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

- `dataset`: Dataset to use ('mnist', 'cifar10', 'cifar100')
- `image_shape`: Shape of the dataset images (default: (32, 32))
- `cl_type`: Type of continual learning ('cil', 'dil')
- `nclasses`: Number of classes in the dataset (default: 10)
- `num_increments`: Number of tasks in continual learning (default: 5)
- `device`: Device for deep learning ('cpu', 'cuda')
- `epoch`: Number of epochs per task (default: 10)
- `batch_size`: Batch size for current data stream (default: 64)
- `buffer_memory_size`: Size of the buffer (default: 500)
- `model`: Continual learning method to use ('er', 'er_ace', 'der', 'der++')
- `buffer_extraction`: Buffer extraction strategy ('random', 'mir')
- `buffer_extraction_size`: Size of the buffer extraction batch (default: 64)
- `buffer_storage`: Buffer storage strategy ('random', 'gss')
- `buffer_storage_size`: Size of the buffer storage update (default: 64)
- `optim`: Optimizer for learning ('sgd')
- `lr`: Learning rate for the optimizer (default: 1e-3)
- `momentum`: Momentum for the optimizer (default: 0.9)
- `alpha`: Hyperparameter for knowledge distillation in DER, DER++ (default: 0.1)
- `beta`: Hyperparameter for knowledge distillation in DER++ (default: 0.5)

### Key Components

- `model/trainer.py`: Manages the training process.
- `model/continual_model.py`: Defines the base continual learning model class.
- `model/er,er_ace,der`: Implements each continual learning method.
- `model/buffer.py`: Implements buffer for exemplar storage and extraction.
- `data/dataloader.py`: Implements data loader for incremental learning.

## Contribution

Hyeonchang Chu [Curriculum Vitae](http://air.cau.ac.kr/)

## License

MIT License

## Contact

email: cngusckd@gmail.com