#include "lorenz.h"

using namespace tODE;

void Lorenz(const state_t& x, state_t& xd)
{
  static constexpr double sigma = 10.0;
  static constexpr double R = 28.0;
  static constexpr double b = 8.0 / 3.0;

  xd[0] = sigma * (x[1] - x[0]);
  xd[1] = R * x[0] - x[1] - x[0] * x[2];
  xd[2] = -b * x[2] + x[0] * x[1];
};

std::vector<state_t> trajectory()
{
   state_t x = { 1.0, 1.0, 1.0 };
   state_t xd = {0.0, 0.0, 0.0 };
   double t = 0.0;
   double dt = 0.01;
   double t_end = 10.0;
   std::vector<state_t> record;
   int step = 0;
   
   //record.reserve(3000);
   while (t < t_end)
   {      
	Lorenz(x, xd);
	for(int i = 0; i < xd.size(); i++){
	    // old derivative becomes new state
	    x[i] += xd[i]*dt;
	}  
	t += dt;
	
   }
   return record;
}


