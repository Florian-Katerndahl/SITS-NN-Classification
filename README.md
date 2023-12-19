# Tree Species Classification with Convolutional Neural Networks using Satellite Image Time Series

This repository contains scripts and modules to train transformer and lst models, means to apply 
trained models for tree species classification in a somewhat portable state by using Poetry 
for dependency and environment management as well as a scalable, portable and maintainable 
Nextflow Workflow.

Not all of the following sections apply to all usecases.

## Installation

:warning: Read the section `tensorflow` before installing from source!

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

# usage: inference.py [-h] -w WEIGHTS --input-tiles INPUT [--input-dir BASE] [--input-glob IGLOB] [--output-dir OUT]
#                     --date-cutoff DATE [--mask-dir MASKS] [--mask-glob MGLOB] [--row-size ROW-BLOCK]
#                     [--col-size COL-BLOCK] [--log] [--log-file LOG-FILE] [--cpus CPUS]
# 
# Run inference with already trained LSTM classifier on a remote-sensing time series represented as FORCE ARD
# datacube.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -w WEIGHTS, --weights WEIGHTS
#                         Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or
#                         absolute file path.
#   --input-tiles INPUT   List of FORCE tiles which should be used for inference. Each line should contain one FORCE
#                         tile specifier (Xdddd_Ydddd).
#   --input-dir BASE      Path to FORCE datacube. By default, use the current PWD.
#   --input-glob IGLOB    Optional glob pattern to restricted files used from `input-dir`.
#   --output-dir OUT      Path to directory into which predictions should be saved. By default, use the current PWD.
#   --date-cutoff DATE    Cutoff date for time series which should be included in datacube for inference.
#   --mask-dir MASKS      Path to directory containing folders in FORCE tile structure storing binary masks with a
#                         value of 1 representing pixels to predict. Others can be nodata or 0. Masking is done on a
#                         row-by-row basis. I.e., the entire unmasked datacube is constructed from the files found in
#                         `input-dir`. Only when handing a row of pixels to the DL-model for inference are data
#                         removed. Thus, this does not reduce the memory footprint, but can speed up inference
#                         significantly under certain conditions.
#   --mask-glob MGLOB     Optional glob pattern to restricted file used from `mask-dir`.
#   --row-size ROW-BLOCK  Row-wise size to read in at once. If not specified, query dataset for block size and assume
#                         constant block sizes across all raster bands in case of multilayer files. Contrary to what
#                         GDAL allows, if the entire raster extent is not evenly divisible by the block size, an error
#                         will be raised and the process aborted. If only `row-size` is given, read the specified
#                         amount of rows and however many columns are given by the datasets block size. If both `row-
#                         size` and `col-size` are given, read tiles of specified size.
#   --col-size COL-BLOCK  Column-wise size to read in at once. If not specified, query dataset for block size and
#                         assume constant block sizes across all raster bands in case of multilayer files. Contrary to
#                         what GDAL allows, if the entire raster extent is not evenly divisible by the block size, an
#                         error will be raised and the process aborted. If only `col-size` is given, read the
#                         specified amount of columns and however many rows are given by the datasets block size. If
#                         both `col-size` and `row-size` are given, read tiles of specified size.
#   --log                 Emit logs?
#   --log-file LOG-FILE   If logging is enabled, write to this file. If omitted, logs are written to stdout.
#   --cpus CPUS           Number of CPUS for Inter-OP and Intra-OP parallelization of pytorch.
```

## Nextflow Workflow Execution

This repository also contains the tree species classification packaged as a [Nextflow](https://www.nextflow.io/) workflow for easy, scalable and 
reproducable execution. For an introduction to Nextflow as well as detailed explanations of the philsophy behind it, 
please vist the follwing websites:

1. https://www.nextflow.io/
2. https://training.nextflow.io/
3. https://www.nextflow.io/docs/latest/index.html

### Prerequisites

#### Installing Nextflow

The official installation instructions are depicted [here](https://www.nextflow.io/docs/latest/getstarted.html#requirements). 

#### Installing Docker

To ease portability, the processes don't run directly on the host operating system. Instead, containers (here: Docker) are 
used to create isolated environments containing (almost) all dependencies needed for execution. Thus, Docker needs to be 
installed on the host system executing the workflow. Please check the [official documentation](https://docs.docker.com/engine/install/) 
on how to install Docker on your system. Additionally, make sure that the user who runs the workflow is allowed to 
use Docker.

### Executing the Pipeline

The workflow is currently setup to be executed either locally, e.g. your personal computer or a single remote 
server, or on a Kubernetes cluster.

The workflow relies on a directory called `bin` in the root of this project which contains additional scripts/binary 
exectuable files which are used in workflow processes. Nextflow automatically adds this folder to the path variable/copies 
the respective files to a remote system and thus makes them usable inside of the containers which are used for workflow 
execution. Symlinking the respective files into this directory is sufficient.

During devlopment or when running the workflow periodically, unsing the caching functionality of Nextflow with the `-resume` 
flag is benefical. Nonetheless, periodically removing the `work` directory (default cache location) and directories created 
under `/tmp` may be necessary.

Any additional workflow configuration is done via the `nextflow.config` file. Specifying parameters on the command line 
during workflow execution will overwrite them.

#### Local Executor

> What may be of interes here?

#### Kubernetes Cluster

*To be implemented*