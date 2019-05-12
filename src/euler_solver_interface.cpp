#include <torch/extension.h>

#include <iostream>
#include <map>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// CUDA declarations

torch::Tensor solver_cuda(torch::Tensor F, torch::Tensor x0, double dt, int steps, int W, std::string name);


// C++ interface

torch::Tensor ode_solve(torch::Tensor F, torch::Tensor x0, double dt, int steps, int W, std::string name){
    CHECK_INPUT(F); 
    CHECK_INPUT(x0);

    return solver_cuda(F, x0, dt, steps, W, name);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solver_cuda", &solver_cuda, "ODE Solver (CUDA)");
}
