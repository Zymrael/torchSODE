#include <torch/extension.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void
euler_kernel(torch::Tensor F, torch::Tensor x0, float dt, int steps, int W) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / W;
    int col = tid % W;

    double x0_in = x0[tid];
    double F_in = F[row][col];
    
    if(index < W*W){
    	for(i = 0; i < steps; i++)
       	   x0 += (F * x0)*dt;
        x0[tid] = x0;
    }
}

torch::Tensor euler_solver_cuda(torch::Tensor F, torch::Tensor x0, double dt, size_t steps, int W){

    // compute number of blocks and threads per block
    
    const int threadsPerBlock = 512;
    const int blocks = (W*W + threadsPerBlock - 1) / threadsPerBlock;

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    // run saxpy_kernel on the GPU
    euler_kernel<<<blocks, threadsPerBlock>>>(F, x0, dt, steps, W);
    cudaDeviceSynchronize();
    return x0;
}
