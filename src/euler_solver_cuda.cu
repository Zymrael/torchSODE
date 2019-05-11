#include <torch/extension.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void
euler_kernel(torch::PackedTensorAccessor<float, 2> F_a, torch::PackedTensorAccessor<float, 1> x0_a, float dt, int steps, int W) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //int row = tid / W;
    //int col = tid % W;
    if(tid < W){
        double x0_in = x0_a[tid];
        double F_in = F_a[tid][tid];
    	for(int i = 0; i < steps; i++)
       	   x0_in += (F_in * x0_in)*dt;
        x0_a[tid] = x0_in;
    }
}

torch::Tensor euler_solver_cuda(torch::Tensor F, torch::Tensor x0, double dt, int steps, int W){
    
    const int threadsPerBlock = 512;
    const int blocks = (W*W + threadsPerBlock - 1) / threadsPerBlock;
    
    auto F_a = F.packed_accessor<float,2>();
    auto x0_a = x0.packed_accessor<float,1>();

    euler_kernel<<<blocks, threadsPerBlock>>>(F_a, x0_a, dt, steps, W);
    cudaDeviceSynchronize();
    return x0;
}
