#include <torch/extension.h>

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <map>

const float A21 = 1.0f/5.0f;
const float C2 = 1.0f/5.0f;

const float A31 = 3.0f/40.0f;
const float A32 = 9.0f/40.0f;
const float C3 = 3.0f/10.0f;

const float A41 = 44.0f/45.0f;
const float A42 = 56.0f/15.0f;
const float A43 = 32.0f/9.0f;
const float C4 = 4.0f/5.0f;

const float A51 = 19372.0f/6561.0f;
const float A52 = 25360.0f/2187.0f;
const float A53 = 64448.0f/6551.0f;
const float A54 = 212.0f/729.0f;
const float C5 = 8.0f/9.0f;

const float A61 = 9017.0f/3168.0f;
const float A62 = 355.0f/33.0f;
const float A63 = 46732.0f/5247.0f;
const float A64 = 49.0f/176.0f;
const float A65 = 5103.0f/18656.0f;

const float A71 = 35.0f/384.0f;
const float A73 = 500.0f/1113.0f;
const float A74 = 125.0f/192.0f;
const float A75 = 2187.0f/6784.0f;
const float A76 = 11.0f/84.0f;

constexpr float B1 = 35.0f/384.0f;
constexpr float B2 = 0.0f;
constexpr float B3 = 500.0f/1113.0f;
constexpr float B4 = 125.0f/192.0f;
constexpr float B5 = 2187.0f/6784.0f;
constexpr float B6 = 11.0f/84.0f;
constexpr float B7 = 0.0f;

constexpr float BS1 = 5179.0f/57600.0f;
constexpr float BS2 = 0.0f;
constexpr float BS3 = 7571.0f/16695.0f;
constexpr float BS4 = 393.0f/640.0f;
constexpr float BS5 = 92097.0f/339200.0f;
constexpr float BS6 = 187.0f/2100.0f;
constexpr float BS7 = 1.0f/40.0f;

constexpr float E1 = B1-BS1;
constexpr float E2 = B2-BS2;
constexpr float E3 = B3-BS3;
constexpr float E4 = B4-BS4;
constexpr float E5 = B5-BS5;
constexpr float E6 = B6-BS6;
constexpr float E7 = B7-BS7;

typedef void (*solver_t)(torch::PackedTensorAccessor<float, 2>, 
		torch::PackedTensorAccessor<float, 1>, 
		torch::PackedTensorAccessor<float, 1>, 
		float, int, int);
typedef float (*method_t)(float, float, float, float, torch::PackedTensorAccessor<float, 1>, int);

template <unsigned int blockSize>
__device__ void parallel_max(torch::PackedTensorAccessor<float, 1> g_idata, float *g_odata, unsigned int n) {
	__shared__ int sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i < n){ sdata[tid] = fmaxf(g_idata[i], g_idata[i+blockSize]);  i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = fmaxf(sdata[tid], sdata[tid +  64]); } __syncthreads(); }

	if (tid < 32){
		if (blockSize >=  64) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
		if (blockSize >=  32) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
		if (blockSize >=  16) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  8]);
		if (blockSize >=   8) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  4]);
		if (blockSize >=   4) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  2]);
		if (blockSize >=   2) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  1]);
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__device__ float
euler_method(float F_in, float x0_in, float g_in, float dt, torch::PackedTensorAccessor<float, 1> err_a, int tid) {
	return (F_in * g_in) * dt;
}

__device__ float
rk4_method(float F_in, float x0_in, float g_in, float dt, torch::PackedTensorAccessor<float, 1> err_a, int tid) {
	auto f1 = (F_in * g_in)*dt;

	auto k2 = dt * f1 / 2.0f;
        auto f2 = (F_in * (g_in + k2)) * (dt / 2.0f);

	auto k3 = dt * f2 / 2.0f;
        auto f3 = (F_in * (g_in + k3)) * (dt / 2.0f);

	auto k4 = dt * f3;
	auto f4 = (F_in * (g_in + k4)) * dt;

	return (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

__device__ float
dopri5_method(float F_in, float x0_in, float g_in, float dt, torch::PackedTensorAccessor<float, 1> err_a, int tid) {
	auto k1 = dt * F_in;
	auto f1 = (F_in * g_in) * dt;
	
	auto k2 = dt * (f1 * A21);
	auto f2 = (F_in * (g_in + k2)) * (dt * C2);

	auto k3 = dt * ((f1 * A31) + (f2 * A32)); 
	auto f3 = (F_in * (g_in + k3)) * (dt * C3);

	auto k4 = dt * ((f1 * A41) - (f2 * A42) + (f3 * A43));
	auto f4 = (F_in * (g_in + k4) * (dt * C4));
        
	auto k5 = dt * ((f1 * A51) - (f2 * A52) + (f3 * A53) - (f4 * A54));
	auto f5 = (F_in * (g_in + k5) * (dt * C5));

	auto k6 = dt * ((f1 * A61) - (f2 * A62) + (f3 * A63) + (f4 * A64) - (f4 * A65));
	auto f6 = (F_in * (g_in + k6) * (dt));

	auto k7 = dt * ((f1 * A71) + (f3 * A73) + (f4 * A74) - (f5 * A75) + (f6 * A76));
	// auto f7 = (F_in * (g_in + k7) * (dt));
	
	auto res = (B1 * f1) + (B3 * f3) + (B4 * f4) - (B5 * f5) + (B6 * f6);  
	auto error = dt * (E1*k1 + E2*k2 + E3*k3 + E4*k4 + E5*k5 + E6*k6 + E7*k7);
	err_a[tid] = error;
	return res;
}

__device__ float
calculate_dt(float dt, float rtol, float maximum_err, float rk_order) {
	return 0.84 * dt * pow (rtol / maximum_err, 1/rk_order); 
}

__device__ float
get_step_size(float dt, float rtol, torch::PackedTensorAccessor<float, 1> err_a, float* max_err, int x0_size) {
	float new_dt = dt;
	float maximum_err = 0;
	for (int i=0; i < blockDim.x; i++) {
		maximum_err = fmaxf(maximum_err, max_err[i]);
	}
	if(maximum_err > rtol) {
		new_dt = calculate_dt(dt, rtol, maximum_err, 5);
	}
	return new_dt;

}

__device__ void
update_dt(int tid, int x0_size, float dt, float rtol, torch::PackedTensorAccessor<float, 1> err_a, float* max_err, float* new_dt) {
    	const int threadsPerBlock = 512; 
	parallel_max<threadsPerBlock>(err_a, max_err, (unsigned int)x0_size);
	__syncthreads();
	if(tid == 0) {
		*new_dt = get_step_size(dt, rtol, err_a, max_err, x0_size);
	}
}

__global__ void
general_solver(method_t method, 
		torch::PackedTensorAccessor<float, 2> F_a, 
		torch::PackedTensorAccessor<float, 1> x0_a, 
		torch::PackedTensorAccessor<float, 1> g_a, 
		torch::PackedTensorAccessor<float, 1> err_a, 
		float dt, int steps, int x0_size, float rtol, float* new_dt, float* max_err, bool isDOPRI) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < x0_size){
        auto x0_in = x0_a[tid];
	auto g_in = g_a[tid];
        auto F_in = F_a[tid][tid];
	float x0_new;

   	for(int i = 0; i < steps; i++) {
		x0_new = method(F_in, x0_in, g_in, dt, err_a, tid);
		x0_in = x0_in + x0_new;
		if(isDOPRI) update_dt(tid, x0_size, dt, rtol, err_a, max_err, new_dt);
	}

        x0_a[tid] = x0_in;
    }
}

__global__ void
compact_diagonal_solver(method_t method, 
		torch::PackedTensorAccessor<float, 2> F_a, 
		torch::PackedTensorAccessor<float, 1> x0_a, 
		torch::PackedTensorAccessor<float, 1> g_a, 
		torch::PackedTensorAccessor<float, 1> err_a, 
		float dt, int steps, int x0_size, float rtol, float* new_dt, float* max_err, bool isDOPRI) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < x0_size){
        auto x0_in = x0_a[tid];
	auto g_in = g_a[tid];
	auto F_in = F_a[0][0];

	float x0_new;
   	for(int i = 0; i < steps; i++) {
		x0_new = method(F_in, x0_in, g_in, dt, err_a, tid);
		x0_in = x0_in + x0_new;
		if(isDOPRI) update_dt(tid, x0_size, dt, rtol, err_a, max_err, new_dt);
	}

        x0_a[tid] = x0_in;
    }
}

__global__ void
general_skew_symmetric_solver(method_t method, 
		torch::PackedTensorAccessor<float, 2> F_a, 
		torch::PackedTensorAccessor<float, 1> x0_a, 
		torch::PackedTensorAccessor<float, 1> g_a, 
		torch::PackedTensorAccessor<float, 1> err_a, 
		float dt, int steps, int x0_size, float rtol, float* new_dt, float* max_err, bool isDOPRI) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < x0_size/2) {
	auto g_in_1 = g_a[tid];
	auto g_in_2 = g_a[tid + x0_size/2];

        auto x0_in_1 = x0_a[tid];
        auto x0_in_2 = x0_a[tid + x0_size/2];

	auto UL_v = F_a[tid][tid];
	auto UR_v = F_a[tid][tid + x0_size/2];
	auto LL_v = F_a[tid + x0_size/2][tid];
	auto LR_v = F_a[tid + x0_size/2][tid + x0_size/2];

	float temp_err = 0;
   	for(int i = 0; i < steps; i++) {
		float x0_in_1_new_1 = method(UL_v, x0_in_1, g_in_1, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);
		float x0_in_1_new_2 = method(UR_v, x0_in_2, g_in_2, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);

		x0_in_1 = x0_in_1 + x0_in_1_new_1 + x0_in_1_new_2;

		float x0_in_2_new_1 = method(LL_v, x0_in_1, g_in_1, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);
		float x0_in_2_new_2 = method(LR_v, x0_in_2, g_in_2, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);

		x0_in_2 = x0_in_1 + x0_in_2_new_1 + x0_in_2_new_2;

		err_a[tid] = temp_err;
		if(isDOPRI) update_dt(tid, x0_size, dt, rtol, err_a, max_err, new_dt);
	}

        x0_a[tid] = x0_in_1;
	x0_a[tid + x0_size/2] = x0_in_2;
    }
}

__global__ void
compact_skew_symmetric_solver(method_t method, 
		torch::PackedTensorAccessor<float, 2> F_a, 
		torch::PackedTensorAccessor<float, 1> x0_a, 
		torch::PackedTensorAccessor<float, 1> g_a, 
		torch::PackedTensorAccessor<float, 1> err_a, 
		float dt, int steps, int x0_size, float rtol, float* new_dt, float* max_err, bool isDOPRI) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < x0_size/2) {
	auto g_in_1 = g_a[tid];
	auto g_in_2 = g_a[tid + x0_size/2];

        auto x0_in_1 = x0_a[tid];
        auto x0_in_2 = x0_a[tid + x0_size/2];

	auto UL_v = F_a[0][0];
	auto UR_v = F_a[0][1];
	auto LL_v = F_a[1][0];
	auto LR_v = F_a[1][1];

	float temp_err = 0;
   	for(int i = 0; i < steps; i++) {
		float x0_in_1_new_1 = method(UL_v, x0_in_1, g_in_1, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);
		float x0_in_1_new_2 = method(UR_v, x0_in_2, g_in_2, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);

		x0_in_1 = x0_in_1 + x0_in_1_new_1 + x0_in_1_new_2;

		float x0_in_2_new_1 = method(LL_v, x0_in_1, g_in_1, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);
		float x0_in_2_new_2 = method(LR_v, x0_in_2, g_in_2, dt, err_a, tid);
		temp_err = fmaxf(temp_err, err_a[tid]);

		x0_in_2 = x0_in_1 + x0_in_2_new_1 + x0_in_2_new_2;

		err_a[tid] = temp_err;
		if(isDOPRI) update_dt(tid, x0_size, dt, rtol, err_a, max_err, new_dt);
	}

        x0_a[tid] = x0_in_1;
	x0_a[tid + x0_size/2] = x0_in_2;
    }
}

// Declare static pointers to device functions
__device__ method_t p_euler_method = euler_method;
__device__ method_t p_rk4_method = rk4_method;
__device__ method_t p_dopri5_method = dopri5_method;

float solve_cuda(torch::Tensor F, torch::Tensor x0, torch::Tensor g, float dt, int steps, std::string name, float rtol, float atol){

    std::map<std::string, method_t> h_methods;
    method_t h_euler_method;
    method_t h_rk4_method; 
    method_t h_dopri5_method; 

    // Copy device function pointers to host side
    cudaMemcpyFromSymbol(&h_euler_method, p_euler_method, sizeof(method_t));
    cudaMemcpyFromSymbol(&h_rk4_method, p_rk4_method, sizeof(method_t));
    cudaMemcpyFromSymbol(&h_dopri5_method, p_dopri5_method, sizeof(method_t));

    h_methods["Euler"] = h_euler_method;
    h_methods["RK4"] = h_rk4_method;
    h_methods["DOPRI5"] = h_dopri5_method;

    method_t d_chosen_method = h_methods[name];

    auto err = torch::zeros_like(x0);

    auto F_a = F.packed_accessor<float,2>();
    auto x0_a = x0.packed_accessor<float,1>();
    auto g_a = g.packed_accessor<float,1>();
    auto err_a = err.packed_accessor<float,1>();

    auto F_size = torch::size(F, 0);
    auto x0_size = torch::size(x0, 0);

    const int threadsPerBlock = 512; 
    const int blocks = (x0_size + threadsPerBlock - 1) / threadsPerBlock;
    float new_dt = dt;
    float* d_new_dt;
    float* h_new_dt = &new_dt;
    cudaMalloc((void **)&d_new_dt, sizeof(float));

    float* max_err;
    cudaMalloc((void **)&max_err, blocks*sizeof(float));
    bool isDOPRI = name == "DOPRI5";

    switch(F_size) {
	case 1:
		compact_diagonal_solver<<<blocks, threadsPerBlock>>>(d_chosen_method, F_a, x0_a, g_a, err_a, dt, steps, x0_size, rtol, d_new_dt, max_err, isDOPRI);
		break;
	case 2:
		compact_skew_symmetric_solver<<<blocks, threadsPerBlock>>>(d_chosen_method, F_a, x0_a, g_a, err_a, dt, steps, x0_size, rtol, d_new_dt, max_err, isDOPRI);
		break;
	default:
		general_skew_symmetric_solver<<<blocks, threadsPerBlock>>>(d_chosen_method, F_a, x0_a, g_a, err_a, dt, steps, x0_size, rtol, d_new_dt, max_err, isDOPRI);
		break;
    }
    
    cudaMemcpy(h_new_dt, d_new_dt, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_new_dt);
    cudaFree(max_err);
    return *h_new_dt;
}

