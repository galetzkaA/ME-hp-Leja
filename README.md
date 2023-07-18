# ME-hp-Leja
Repository contains main source code for an hp-adaptive multi-element stochastic collocation method based on Leja nodes. The algorithm allows re-use existing model evaluations during either h- or p-refinement. The collocation method is based on weighted Leja nodes. After h-refinement, local interpolations are stabilized by adding and sorting Leja nodes on each newly created sub-element in a hierarchical manner. For p-refinement, the local polynomial approximations are based on total-degree bases.

```
@article{galetzka2023,
author = {Galetzka, Armin and Loukrezis, Dimitrios and Georg, Niklas and De Gersem, Herbert and Römer, Ulrich},
title = {An hp-adaptive multi-element stochastic collocation method for surrogate modeling with information re-use},
journal = {International Journal for Numerical Methods in Engineering},
volume = {124},
number = {12},
pages = {2902-2930},
doi = {https://doi.org/10.1002/nme.7234},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.7234},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.7234},
year = {2023}
}
```

## Content

- This is an algorithm for the efficient surrogate modeling for objective functions that feature non-smooth or strongly localised response surfaces.

- The method can be applied in the context of forward and inverse uncertainty quantification.

- The weighted Leja nodes allow for arbitrary parameter distributions.

- Best for problems with low to moderate ($O$(10)) dimensionality.

## Running the examples
The example folder contains three examples with varying regularity of the objective function.

The main files to run are ```continuous_2D.py```, ```reduced_regularity_2D.py``` and ```discontinuous_2D.py```


## Licence

This project is licensed under the terms of the GNU General Public License (GPL).
