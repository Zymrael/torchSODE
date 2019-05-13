#include <torch/extension.h>

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <map>

typedef void (*solver_t)(torch::PackedTensorAccessor<float, 2>, torch::PackedTensorAccessor<float, 1>, float, int, int);
typedef void (*method_t)(double, double, float, int);

typedef std::string string;
typedef std::map<string, method_t> map;

__inline__ __global__ void
euler_method(double F_in, double x0_in, float dt, int steps) {
       	x0_in += (F_in * x0_in)*dt;
}

__inline__ __global__ void
rk4_method(double F_in, double x0_in, float dt, int steps) {
	auto f1 = (F_in * x0_in)*dt;

	auto c2 = dt * f1 / 2.0;
        auto f2 = (F_in * (x0_in + c2)) * (dt / 2.0);

	auto c3 = dt * f2 / 2.0;
        auto f3 = (F_in * (x0_in + c3)) * (dt / 2.0);

	auto c4 = dt * f3;
	auto f4 = (F_in * (x0_in + c4)) * dt;

	x0_in = x0_in + (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0;
}

__global__ void
general_solver(method_t method, torch::PackedTensorAccessor<float, 2> F_a, torch::PackedTensorAccessor<float, 1> x0_a, float dt, int steps, int W) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < W){
        double x0_in = x0_a[tid];
	
        double F_in = F_a[tid][tid];

   	for(int i = 0; i < steps; i++) {
		method(F_in, x0_in, dt, steps);
	}

        x0_a[tid] = x0_in;
    }
}

torch::Tensor solver_cuda(torch::Tensor F, torch::Tensor x0, double dt, int steps, int W, string name){

    map methods;
    methods["Euler"] = euler_method;    
    methods["RK4"] = rk4_method;
    method_t chosen_method = methods[name];

    auto F_a = F.packed_accessor<float,2>();
    auto x0_a = x0.packed_accessor<float,1>();
    auto F_size = F_a::size;

    auto xud = torch::chunk(x1, 2, 0);
    auto xulr = torch::chunk(xud[0], 2, 1);
    auto xllr = torch::chunk(xud[1], 2, 1);

/*
    auto UL = xulr[0].packed_accessor<float, 2>();
    auto UR = xulr[1].packed_accessor<float, 2>();
    auto LL = xllr[0].packed_accessor<float, 2>();
    auto LR = xllr[1].packed_accessor<float, 2>();
*/

    /*if(F_a == F_a[0][0] * torch::eye(F_size)) {
    	const int threadsPerBlock = 512;
    	const int blocks = (W + threadsPerBlock - 1) / threadsPerBlock;	

    } else if(	F_a[0] == UL 				&& F_a[F_size/2][0] == LL && 
		UR[0][0]*torch::eye(F_size/2) == UR 	&& F_a[F_size/2][F_size/2] == LR)
	// Launch UL, LL, UR, LR kernel
    	const int threadsPerBlock = 512;
    	const int blocks = (W + threadsPerBlock - 1) / threadsPerBlock;

    } else {*/
    	const int threadsPerBlock = 512;
    	const int blocks = (W*W + threadsPerBlock - 1) / threadsPerBlock;
	general_solver<<<blocks, threadsPerBlock>>>(chosen_method, F_a, x0_a, dt, steps, W, method);
    //}

    cudaDeviceSynchronize();
    return x0;
}


