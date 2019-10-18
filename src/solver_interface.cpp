#include <torch/extension.h>

#include <iostream>
#include <map>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA declarations

float solve_cuda(torch::Tensor F, torch::Tensor x0, torch::Tensor g, float dt, int steps, std::string name, float rtol=1e-3, float atol=1e-6);


// C++ interface

float solve_cpp(torch::Tensor F, torch::Tensor x0, torch::Tensor g, float dt, int steps, std::string name, float rtol=1e-3, float atol=1e-6){
    CHECK_INPUT(F); 
    CHECK_INPUT(x0);
    CHECK_INPUT(g);

    return solve_cuda(F, x0, g, dt, steps, name, rtol, atol);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve", &solve_cpp, "ODE Solver (CUDA)");
}
