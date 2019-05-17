
# torchSODE

CUDA solver callable from PyTorch. Optimized for independent ordinary differential equations (ODEs) that can be represented as a sparse block diagonal matrix. 

The solver itself is designed to be used during neural network training and thus accepts an additional argument `grad`

## Installation

In your terminal of choice:

`git clone https://github.com/Zymrael/torchODE.git`

`cd torchSODE`

`python setup.py install`


In your python files or notebooks:

`import torchSODE`

## API
`torchSODE.solve(F, x0, grad, dt, steps, method='Euler')` performs `steps` integration cycle with `dt` step size. 

For problems where the size of x0 is too large allocating a matrix of dimensions size * size is not always possible. In these cases we assume a compressed representation of `F` which exposes only its diagonal values.

The following convention is used (regardless of problem size):
1. For diagonal `F` allocate a torch.Tensor of shape (1).
2. For 4 block-diagonal `F`, allocate a torch.Tensor of shape (2,2) with values of upper-left diagonal in position [0,0], upper-right diagonal [0,1], lower-left diagonal [1,0], lower-right diagonal [1,1].

In any other scenario `torchSODE.solve` requires the size of `x0.size(0)` to match `F.size(0)` and `F.size(1)`, with `F` is assumed to be 4 block-diagonal.

### Methods
`'Euler'` = Euler

`'RK4'` = Runge-Kutta 4
