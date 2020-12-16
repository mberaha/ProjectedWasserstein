# Code for '' Projected Statistical Methods for Distributional Data on the Real Line with the Wasserstein Metric ''

## Requirements
The code was developed and tested on an Ubuntu 18.04 machine with the following libraries

```
python 3.8
numpy==1.18.2
scipy==1.4.1
scikit-learn==0.22.2.post1
pyomo==5.7
qpsolvers==1.1
```

Moreover, the optimization library Ipopt (version 3.12.12) was used
and called via pyomo.
To install ipopt, follow the instructions at 
https://coin-or.github.io/Ipopt/INSTALL.html
The Ipopt solver is used only to compute the global and nested geodesic PCA, not
the projected one.

## Directory structure
The actual implementation of the algorithms and some utilities is found
in the folder `pwass`

The folder `scripts` contain python scripts to run the simulated examples.
For instance, to run the Bernstein simulation, it is sufficient to run
from the terminal (from the current / root folder):
```
python3 -m scripts.bernstein_simulation
```
Default arguments used in the simulations are given for all the command line
arguments.

The folder `data` contains the Covid-19 deaths dataset and the Wind speed dataset.

Finally in the current folder some Jupyter Notebooks are present, to 
reproduce all the plots in the manuscript and to run some of the simulations

