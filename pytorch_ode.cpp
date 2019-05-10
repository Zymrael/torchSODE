#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

#include <vector>

typedef torch::Tensor (*model_t)(torch::Tensor, torch::Tensor, double);
typedef torch::Tensor (*solver_t)(size_t, torch::Tensor, torch::Tensor, double, model_t);

torch::Tensor genericModel(torch::Tensor F, torch::Tensor x, double dt) {
	return torch::matmul(F, x) * dt;
}

torch::Tensor euler_solver(size_t t0, torch::Tensor F, torch::Tensor x0, double dt, model_t mdl) {
	auto dxdt = mdl(F, x0, dt);
	x0 = x0 + dxdt;
	return x0;
}

torch::Tensor rk4(size_t t0, torch::Tensor F, torch::Tensor x0, double dt, model_t mdl) {
	auto f1 = mdl(F, x0, dt);

	auto c2 = dt * f1 / 2.0;
	auto f2 = mdl(F, x0 + c2, dt/2.0);

	auto c3 = dt * f2 / 2.0;
	auto f3 = mdl(F, x0 + c3, dt/2.0);

	auto c4 = dt * f3;
	auto f4 = mdl(F, x0 + c4, dt);

	x0 = x0 + (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0;
	return x0;
}

std::vector<torch::Tensor> ode_solver(torch::Tensor F, torch::Tensor x0) {
	model_t model = &genericModel;
	solver_t solver = &euler_solver;
	double dt = 0.01;
	size_t range = 1000;

	torch::Tensor initial_x0 = x0;
	torch::Tensor x_new;

	for(size_t t = 0; t < range; t++) {
		x_new = solver(0, F, x0, dt, model);
		// Append traj? I guess we don't need this for the final implementation		
		x0 = x_new;
	}
	return {initial_x0, x_new};
}

/*
std::vector<at::Tensor> lltm_forward(
    torch::Tensor old_cell) {

*/  
//  auto X = torch::cat({old_h, input}, /*dim=*/1);
/*  torch::matmul(F, x) * dt;

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
*/
//  auto gates = gate_weights.chunk(3, /*dim=*/1);
/*
  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
*/
//  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

/*
  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;
*/
//  auto gates = gate_weights.chunk(3, /*dim=*/1);
/*  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
*/
//      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

//  auto d_weights = d_gates.t().mm(X);
//  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);
/*
  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
*/
//auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  //auto d_input = d_X.slice(/*dim=*/1, state_size);

//  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
//}
