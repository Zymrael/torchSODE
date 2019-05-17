#include <torch/extension.h>

#include <iostream>
#include <map>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

#include <vector>


typedef std::string string;
typedef void (*model_t)(torch::Tensor, torch::Tensor, float);
typedef void (*solver_t)(size_t, torch::Tensor, torch::Tensor, float, model_t);
typedef std::map<string, solver_t> map;

void genericModel(torch::Tensor F, torch::Tensor x, float dt) {
	torch::matmul(F, x) * dt;
}

void euler_solver(size_t t0, torch::Tensor F, torch::Tensor x0, float dt, model_t mdl) {
	auto dxdt = mdl(F, x0, dt);
	x0 = x0 + dxdt;
}

void rk4_solver(size_t t0, torch::Tensor F, torch::Tensor x0, float dt, model_t mdl) {
	auto f1 = mdl(F, x0, dt);

	auto c2 = dt * f1 / 2.0;
	auto f2 = mdl(F, x0 + c2, dt/2.0);

	auto c3 = dt * f2 / 2.0;
	auto f3 = mdl(F, x0 + c3, dt/2.0);

	auto c4 = dt * f3;
	auto f4 = mdl(F, x0 + c4, dt);

	x0 = x0 + (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0;
}

void solve(torch::Tensor F, torch::Tensor x0, float dt, size_t steps, string solver) {
	/**
	* Solver for linear systems of independent differential equations.
	*
	*@F torch.Tensor containing the dynamics of the system
	*@x0 torch.Tensor with initial conditions
	*@dt float time delta of each step
	*@steps int number of integration steps 
	*/
	map solvers;

	solvers["Euler"] = euler_solver;
	solvers["RK4"] = rk4_solver;

	solver_t chosen_solver = solvers[solver];

	model_t model = &genericModel;
	torch::Tensor x_new = x0;

	for(size_t t = 0; t < steps; t++) {
		chosen_solver(0, F, x_new, dt, model);
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve", &solve, "ODE Solver (CPP)");
}

