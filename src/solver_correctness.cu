#include <torch/extension.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <map>

const float A21 = 1.0f/5.0f;

const float A31 = 3.0f/40.0f;
const float A32 = 9.0f/40.0f;

const float A41 = 44.0f/45.0f;
const float A42 = 56.0f/15.0f;
const float A43 = 32.0f/9.0f;

const float A51 = 19372.0f/6561.0f;
const float A52 = 25360.0f/2187.0f;
const float A53 = 64448.0f/6551.0f;
const float A54 = 212.0f/729.0f;

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

const float B1 = 35.0f/384.0f;
const float B3 = 500.0f/1113.0f;
const float B4 = 125.0f/192.0f;
const float B5 = 2187.0f/6784.0f;
const float B6 = 11.0f/84.0f;
const float B7 = 0.0f;

const float BS1 = 5179.0f/57600.0f;
const float BS3 = 7571.0f/16695.0f;
const float BS4 = 393.0f/640.0f;
const float BS5 = 92097.0f/339200.0f;
const float BS6 = 187.0f/2100.0f;
const float BS7 = 1.0f/40.0f;

const float E1 = B1-BS1;
const float E3 = B3-BS3;
const float E4 = B4-BS4;
const float E5 = B5-BS5;
const float E6 = B6-BS6;
const float E7 = B7-BS7;

typedef void (*solver_t)(torch::PackedTensorAccessor<float, 2>, 
		torch::PackedTensorAccessor<float, 1>, 
		torch::PackedTensorAccessor<float, 1>, 
		float, int, int);
// typedef float (*method_t)(float, float, float, float, torch::PackedTensorAccessor<float, 1>, int);

typedef float* (*method_t)(float, float, torch::PackedTensorAccessor<float, 1>, torch::PackedTensorAccessor<float, 1>);

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

/*__device__ float
euler_method(float F_in, float x0_in, float g_in, float dt, torch::PackedTensorAccessor<float, 1> err_a, int tid) {
	return (F_in * g_in) * dt;
}*/


__device__ float*
van_der_pol(torch::PackedTensorAccessor<float, 1> g_a, float* k) {
	float y_0 = g_a[0] + k[0];
	float y_1 = g_a[1] + k[1];
	int mu = 100;
	float result[2];
	result[0] = y_1;
	result[1] = mu*(1-pow(y_0, 2))*y_1 - y_0;
	return result;
}

__device__ float*
euler_method(float F_in, float dt, torch::PackedTensorAccessor<float, 1> g_a, torch::PackedTensorAccessor<float, 1> err_a) {
	float empty[2] = {0};
	float* k1 = van_der_pol(g_a, empty);
	float result[2];
	result[0] = dt * k1[0];
	result[1] = dt * k1[1];
	return result;
}

__device__ float*
vector_multiply(float* k, float* k1, float c) {
	float res[2];
	if(c != 0) {
		res[0] = k[0] * c;
		res[1] = k[1] * c;
	}
	res[0] = k[0] * k1[0];
	res[1] = k[1] * k1[1];
	return res;	
}

__device__ float*
vector_sub(float* k, float* k1, float c) {
	float res[2];
	if(c != 0) {
		res[0] = k[0] - c;
		res[1] = k[1] - c;
	}
	res[0] = k[0] - k1[0];
	res[1] = k[1] - k1[1];
	return res;
}

__device__ float*
vector_add(float* k, float* k1, float c) {
	float res[2];
	if(c != 0) {
		res[0] = k[0] + c;
		res[1] = k[1] + c;
	}
	res[0] = k[0] + k1[0];
	res[1] = k[1] + k1[1];
	return res;
}


__device__ float*
rk4_method(float F_in, float dt, torch::PackedTensorAccessor<float, 1> g_a, torch::PackedTensorAccessor<float, 1> err_a) {
	float empty[2] = {0};
	float* k1 = van_der_pol(g_a, empty); 
	float* k2 = van_der_pol(g_a, vector_multiply(
				vector_multiply(k1, empty, 1/2.0f), empty, dt));
	float* k3 = van_der_pol(g_a, vector_multiply(
				vector_multiply(k2, empty, 1/2.0f), empty, dt));
	float* k4 = van_der_pol(g_a, vector_multiply(k3, empty, dt));
	float result[2];
	result[0] = dt * ((k1[0] + (2.0f * k2[0]) + (2.0f * k3[0]) + k4[0]) / 6.0f);
	result[1] = dt * ((k1[1] + (2.0f * k2[1]) + (2.0f * k3[1]) + k4[1]) / 6.0f);

	return result;
}

__device__ float*
dopri5_method(float F_in, float dt, torch::PackedTensorAccessor<float, 1> g_a, torch::PackedTensorAccessor<float, 1> err_a) {
	float empty[2] = {0};
	float* k1 = van_der_pol(g_a, empty);
	float* k2 = van_der_pol(g_a, vector_multiply(
				vector_multiply(k1, empty, A21), empty, dt));
	float* k3 = van_der_pol(g_a,
			vector_multiply(
				vector_add(
					vector_multiply(k1, empty, A31), 
					vector_multiply(k2, empty, A32),
					0
				),
		       		empty,
				dt
			)
		);
	float* k4 = van_der_pol(g_a,
			vector_multiply(
		       		vector_add(	
					vector_sub(
						vector_multiply(k1, empty, A41), 
						vector_multiply(k2, empty, A42),
						0
					),
					vector_multiply(k3, empty, A43),
					0
				), 
		       		empty, 
		       		dt
			)
		);
		
	float* k5 = van_der_pol(g_a,
			vector_multiply(
		       		vector_add(	
					vector_sub(
						vector_multiply(k1, empty, A51), 
						vector_multiply(k2, empty, A52),
						0
					),
					vector_sub(
						vector_multiply(k3, empty, A53),
						vector_multiply(k4, empty, A54),
						0
					),
					0
				), 
		       		empty, 
		       		dt
			)
		);

	float* k6 = van_der_pol(g_a,
			vector_multiply(
				vector_sub(
		       			vector_add(	
						vector_sub(
							vector_multiply(k1, empty, A61), 
							vector_multiply(k2, empty, A62),
							0
						),
						vector_add(
							vector_multiply(k3, empty, A63),
							vector_multiply(k4, empty, A64),
							0
						),
						0
					),
					vector_multiply(k5, empty, A65),
					0
				), 
				empty, 
				dt
			)
		);

	float* k7 = van_der_pol(g_a,
			vector_multiply(
				vector_add(
		       			vector_add(	
						vector_add(
							vector_multiply(k1, empty, A71), 
							vector_multiply(k3, empty, A73),
							0
						),
						vector_sub(
							vector_multiply(k4, empty, A74),
							vector_multiply(k5, empty, A75),
							0
						),
						0
					),
					vector_multiply(k6, empty, A76),
					0
				), 
				empty, 
				dt
			)
		);
	

	err_a[0] = dt * (k1[0]*E1 + k3[0]*E3 + k4[0]*E4 - k5[0]*E5 + k6[0]*E6 + k7[0]*E7);
	err_a[1] = dt * (k1[1]*E1 + k3[1]*E3 + k4[1]*E4 - k5[1]*E5 + k6[1]*E6 + k7[1]*E7);

	float result[2];
	result[0] = dt * (k1[0]*B1 + k3[0]*B3 + k4[0]*B4 - k5[0]*B5 + k6[0]*B6);
	result[1] = dt * (k1[1]*B1 + k3[1]*B3 + k4[1]*B4 - k5[1]*B5 + k6[1]*B6);
	return result;
}

/* __device__ float
dopri5_method(float F_in, float x0_in, float g_in, float dt, torch::PackedTensorAccessor<float, 1> err_a, int tid) {
	float empty[2] = {0};
	float* k1 = array_transform(van_der_pol(g_a, empty), F_in, dt); 

	auto k1 = dt * (F_in * g_in);
	auto k2 = dt * (F_in * g_in * (k1 * A21));
	auto k3 = dt * (F_in * g_in * ((k1 * A31) + (k2 * A32))); 
	auto k4 = dt * (F_in * g_in * ((k1 * A41) - (k2 * A42) + (k3 * A43)));
	auto k5 = dt * (F_in * g_in * ((k1 * A51) - (k2 * A52) + (k3 * A53) - (k4 * A54)));
	auto k6 = dt * (F_in * g_in * ((k1 * A61) - (k2 * A62) + (k3 * A63) + (k4 * A64) - (k5 * A65)));
	auto k7 = dt * (F_in * g_in * ((k1 * A71) 	       + (k3 * A73) + (k4 * A74) - (k5 * A75) + (k6 * A76)));
	
	auto res = (B1 * k1) + (B3 * k3) + (B4 * k4) - (B5 * k5) + (B6 * k6);  
	auto error = dt * (E1*k1 + E3*k3 + E4*k4 + E5*k5 + E6*k6 + E7*k7);
	err_a[tid] = error;
	return res;
} */

/*__device__ float
some_method() {
	thrust::device_vector<float> D(4);
	return 0.0f;
}*/

/*__device__ float
rk4_method(float F_in, float x0_in, float g_in, float dt, torch::PackedTensorAccessor<float, 1> err_a, int tid) {
	auto k1 = dt * (F_in * g_in);
        auto k2 = dt * (F_in * (g_in + (k1/2.0f)));
        auto k3 = dt * (F_in * (g_in + (k2/2.0f)));
	auto k4 = dt * (F_in * (g_in + k3));

	return (k1 + (2.0f * k2) + (2.0f * k3) + k4) / 6.0f;
}*/

__device__ float
calculate_dt(float dt, float tol, float maximum_err, float rk_order) {
	return 0.84 * dt * pow (tol / maximum_err, 1/rk_order); 
}

__device__ float
get_max(int x0_size, torch::PackedTensorAccessor<float, 1> err_a) {
    	// const int threadsPerBlock = 512; 
	// int threads = min(x0_size, threadsPerBlock);
	// int blocks = 1 + (x0_size/threadsPerBlock);
	float maximum_err = fmaxf(err_a[0], err_a[1]);
	//parallel_max<2>(err_a, max_err, (unsigned int)x0_size);
	// __syncthreads();
	/*float maximum_err = 0.0f;
	for (int i=0; i < blockDim.x; i++) {
		maximum_err = fmaxf(maximum_err, max_err[i]);
	}*/
	return maximum_err;
}

/*__global__ void
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
}*/

__global__ void
compact_diagonal_solver(method_t method, 
		torch::PackedTensorAccessor<float, 2> F_a, 
		torch::PackedTensorAccessor<float, 1> x0_a, 
		torch::PackedTensorAccessor<float, 1> g_a, 
		torch::PackedTensorAccessor<float, 1> err_a, 
		float dt, int steps, int x0_size, float tol, float* new_dt, float* max_err, bool isDOPRI) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //if(tid < x0_size){
    if (dt <= 0) {
	    return;
    }
    if(tid==0) {
        //auto x0_in = x0_a[tid];
	float x0_in_0 = x0_a[0];
	float x0_in_1 = x0_a[1];

	// auto g_in = g_a[tid];
	auto F_in = F_a[0][0];

	float* x0_new;
	float* temp_x0_new;

   	for(int i = 0; i < steps; i++) {
		x0_new = method(F_in, dt, g_a, err_a);
		// x0_new = method(F_in, x0_in, g_in, dt, err_a, tid);
		//x0_in = x0_in + x0_new;
		if(tid==0 && isDOPRI) {
			float maximum_error = get_max(x0_size, err_a);
			if(maximum_error > tol) {
				temp_x0_new = x0_new;
				while (maximum_error > tol) {
				  	if(dt == 0) return;
					dt = calculate_dt(dt, tol, maximum_error, 5);
					*new_dt = dt;
					temp_x0_new = method(F_in, dt, g_a, err_a);
					maximum_error = get_max(x0_size, err_a);
				}
				dt = calculate_dt(dt, tol, maximum_error, 5);
				*new_dt = dt;
				x0_new = temp_x0_new;
			} else {
				dt = calculate_dt(dt, tol, maximum_error, 5);
				*new_dt = dt;
			}
		}

		x0_in_0 = x0_in_0 + x0_new[0];
		x0_in_1 = x0_in_1 + x0_new[1];

		g_a[0] = x0_in_0;
		g_a[1] = x0_in_1;
	}

        //x0_a[tid] = x0_in;
	x0_a[0] = x0_in_0;
	x0_a[1] = x0_in_1;
    }
}
/*
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
}*/

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
	default:
//	case 1:
		compact_diagonal_solver<<<blocks, threadsPerBlock>>>(d_chosen_method, F_a, x0_a, g_a, err_a, dt, steps, x0_size, rtol, d_new_dt, max_err, isDOPRI);
		break;
/*	case 2:
		compact_skew_symmetric_solver<<<blocks, threadsPerBlock>>>(d_chosen_method, F_a, x0_a, g_a, err_a, dt, steps, x0_size, rtol, d_new_dt, max_err, isDOPRI);
		break;
	default:
		general_skew_symmetric_solver<<<blocks, threadsPerBlock>>>(d_chosen_method, F_a, x0_a, g_a, err_a, dt, steps, x0_size, rtol, d_new_dt, max_err, isDOPRI);
		break;
		*/
    }
    
    cudaMemcpy(h_new_dt, d_new_dt, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_new_dt);
    cudaFree(max_err);
    return *h_new_dt;
}

