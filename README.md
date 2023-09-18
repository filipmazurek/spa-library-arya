# SPA

SMC for Processor Analysis.

SPA is a statistics framework which allows users to use Statistical Model Checking (SMC)-based techniques to analyze numerical results. The technique is described in the paper [Rigorous Evaluation of Computer Processors with Statistical Model Checking](https://doi.org/10.1145/3613424.3623785).
## Installation

### Conda: Installation Without Source (Recommended)

The file `spa_env_no_source.yml` provides a Conda environment file to set up the environment containing the SPA 
library. 
The following code will
1. Download the required environment file
2. Create a new Conda environment called `spa`

```shell
$ wget https://raw.githubusercontent.com/filipmazurek/spa-library/v0.0.1/spa_env_no_source.yml
$ conda env create -f spa_env_no_source.yml
$ conda activate spa
```

### Conda: Installation From Source

This is recommended for further development of the SPA library.

We provided a Conda environment file to set up the correct environment.

```shell
$ git clone --depth 1 --branch v0.0.1 git@github.com:filipmazurek/spa-library.git
$ cd spa-library/
$ conda env create -f environment.yml
$ conda activate spa
```

This will install the SPA library in editable mode, so that changes in the repository will reflect in the installation.

## Usage
Please refer to `examples/spa-example.py` for recommended usage of the SPA library, particularly the function `property_use()`. The main use of this library is to create confidence intervals.
