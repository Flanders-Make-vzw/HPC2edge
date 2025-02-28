# HPC2edge Neural Networks

Neural networks module of HPC2edge.

## Get started

Requirements:
- Python3 with virtual-env support

Run the following command the install the dependencies:
```bash
./install.sh
```

## Training

To start training using a predefined configuration:
```bash
./train.sh <config filepath>
```
Example configuration files are available in [`configs/`](configs/).

If you want to embed training in another script:
```python
from nn import training
from nn import utils

config = utils.config(<your config dictionnary>)
training.train(config)
```

The configuration dictionnary must have a similar structure as the example configuration files in [`configs/`](configs/). We therefore recommend first importing one of them as a blueprint and then modify the necessary values.