#include <torch/extension.h>

#include <iostream>
#include <map>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

typedef std::string string;

// CUDA declarations

void solve_cudaa(torch::Tensor F, torch::Tensor x0, torch::Tensor g, float dt, int steps, string name);


// C++ interface

void solve_cpp(torch::Tensor F, torch::Tensor x0, torch::Tensor g, float dt, int steps, string name){
    CHECK_INPUT(F); 
    CHECK_INPUT(x0);
    CHECK_INPUT(g);

    solve_cudaa(F, x0, g, dt, steps, name);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve_cpp", &solve_cpp, "ODE sdadsadsasadsadadsSolver (CUDA)");
}
