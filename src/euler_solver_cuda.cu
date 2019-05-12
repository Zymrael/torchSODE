#include <torch/extension.h>

#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <map>

typedef std::string string;
typedef void (*solver_t)(torch::Tensor, torch::Tensor, double, int, int);
typedef std::map<string, solver_t> map;

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

__global__ void
rk4_kernel(torch::PackedTensorAccessor<float, 2> F_a, torch::PackedTensorAccessor<float, 1> x0_a, float dt, int steps, int W) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < W){
        double x0_in = x0_a[tid];
        double F_in = F_a[tid][tid];

    	for(int i = 0; i < steps; i++){
		auto f1 = (F_in * x0_in)*dt;

		auto c2 = dt * f1 / 2.0;
                auto f2 = (F_in * (x0_in + c2)) * (dt / 2.0);

		auto c3 = dt * f2 / 2.0;
                auto f3 = (F_in * (x0_in + c3)) * (dt / 2.0);

		auto c4 = dt * f3;
		auto f4 = (F_in * (x0_in + c4)) * dt;

		x0_in = x0_in + (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0;
	}
        x0_a[tid] = x0_in;
    }
}


torch::Tensor solver_cuda(torch::Tensor F, torch::Tensor x0, double dt, int steps, int W, string name){

    map solvers;
    solvers["Euler"] = euler_kernel;    
    solvers["RK4"] = rk4_kernel;
    solver_t chosen_kernel_solver = solvers[name];

    const int threadsPerBlock = 512;
    const int blocks = (W*W + threadsPerBlock - 1) / threadsPerBlock;
    
    auto F_a = F.packed_accessor<float,2>();
    auto x0_a = x0.packed_accessor<float,1>();

   



    chosen_kernel_solver<<<blocks, threadsPerBlock>>>(F_a, x0_a, dt, steps, W);
    cudaDeviceSynchronize();
    return x0;
}


