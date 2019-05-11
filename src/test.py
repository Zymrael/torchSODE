import torch
import pySODE

if __name__ == '__main__':
	x_orig = 10*torch.rand(10)
	x0 = x_orig.cuda()
	F = torch.eye(10)
	F = -0.5*F
	F = F.cuda()
	traj2 = []
	dt = 0.1
	for t in range(10):
	    x_new = pySODE.eulerCUDA(F, x0, dt, 1, 10)
	    traj2.append(x_new)
	    x0 = x_new

