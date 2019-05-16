
# torchSODE

CUDA solver callable from PyTorch. Optimized for independent ordinary differential equations (ODEs) that can be represented as a sparse block diagonal matrix. 

The solver itself is designed to be used during neural network training and thus accepts an additional argument `grad`


## API
`torchSODE.solve(F, x0, grad, dt, step, size, method='Euler')`

### Methods
`'Euler'` = Euler
`'RK4' = Runge-Kutta 4
