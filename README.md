# TSC_CNN
Convolutional neural network for tree species classification

## Installation

> :warning: Read the section `tensorflow` before installing from source!

### Python Wheels

**Currently not available!**

### From Repository

#### Pytorch

Unfortunately, the dependency management poetry offers makes the installation of pytorch somewhat cumbersome. By default,
the CUDA 12.1 versions of Pytorch installed. Should you want to install other versions (i.e. CPU wheels or CUDA 11.8),
the following commands are necessary after installation:

```bash
poetry install

# for CUDA 11.8
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cu118 torch torchvision torchaudio

# for CPU wheels
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cpu torch torchvision torchaudio
```

To revert back to the CUDA 12.1 wheels, run:

```bash
poetry remove torch torchvision torchaudio
poetry add torch torchvision torchaudio
```

#### Tensorflow

Apparently, since tensorflow 2.11, the metadata content in the supplied wheels differ from platform to platform insted
of using version markers. Because of that, installation using poetry fails since it downloads the first wheel it
finds. There are two possible soultions, while I only managed to get things working using the first:

1. Specify a specific verison of tensorflow, e.g. `poetry add tensorflow==2.15.0` will install the newest version at
the time of writing
2. Apply patches to poetry before installing the dependencies as specified [here](https://github.com/mazyod/poetry-legacy-index).

## Usage

### Standalone Scripts

Tree species can be predicted with the standalone `inference.py` script. Currently, inference is possible with **LSTM 
classifier only**. Please note, that a [FORCE](https://force-eo.readthedocs.io/en/latest/) datacube is expected as input.
If you installed `sits_classifier` by cloning this repository and running `poetry install`, you must work within the 
poetry shell which masks the python interpreter. All other installed system binaries are still available to you. 

> Note that other environment managers such as conda should probably be quit beforehand. Thus, running e.g. 
> `conda deactivate` is suggested. 

```bash
poetry shell

python apps/inference.py --help

UPDATE HELP OUTPUT!!!
```

## Nextflow Workflow Execution

This repository also contains the tree classification packaged as a [Nextflow](https://www.nextflow.io/) workflow for easy, scalable and 
reproducable execution. For an introduction to Nextflow as well as detailed explanations of the philsophy behind it, 
please vist the follwing websites:

1. https://www.nextflow.io/
2. https://training.nextflow.io/
3. https://www.nextflow.io/docs/latest/index.html

### Prerequisites

#### Installing Nextflow

#### Installing Docker

### Executing the Pipeline

- mention bin folder (symlink to apps currently)
- with resume, without

#### Local Executor

#### Kubernetes Cluster
