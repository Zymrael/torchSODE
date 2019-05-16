
# torchSODE

CUDA solver callable from PyTorch. Optimized for independent ordinary differential equations (ODEs) that can be represented as a sparse block diagonal matrix. 

The solver itself is designed to be used during the process of neural network training and thus accepts an additional argument `grad`

## API
`torchSODE.solve(F, x0, grad, dt, step, size, method='Euler')`
