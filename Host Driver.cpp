#include <cmath>

#define UROUND (2.22e-16)
#define SAFETY 0.9
#define PGROW (-0.2)
#define PSHRNK (-0.25)
#define ERRCON (1.89e-4)
#define TINY(1.0e-30)
const double eps = 1.0e-10;



double * yHost;
yHost = (double *) malloc (numODE * NEQN * sizeof(double));
gHost = (double *) malloc (numODE * NEQN * sizeof(double));

// column-major ordering: variables of the same ODE will be stored in columns of yHost
// Initial conditions for the various ODEs
for (int = 0; i < numODE; ++i) {
    for ( int j = 0; j < NEQN; ++j) {
        yHost[i + numODE * j] = y[i][j];
        gHost[i + numODE * j] = g[i][j];
    }
}

double *yDevice;
cudaMalloc((void**)  &yDevice, numODE * NEQN * sizeof(double));


// heuristic choice of block size based on total num of ODEs
// should be tuned for performance
int thresh_low = std::pow(2, 22);
int thresh_med = std::pow(2, 23);
int thresh_high = std::pow(2, 24);

int blockSize;
if (numODE < thresh_low){
    blockSize = 64;
} else if (numODE < thresh_med){
    blockSize = 128;
} else if (numODE < thresh_high){
    blockSize = 256;
} else {
    blockSize = 512;
}
dim3 dimBlock (blockSize, 1);
dim3 dimGrid (numOde / dimBlock.x, 1)

double t = t0;
double tNext = t + h;

while(t < tEnd){
    cudaMemcpy(yDevice, yHost, numODE*NEQN*sizeof(double), cudaMemcpyHostToDevice);
    // integration kernel
    // "restart" integration style: no information passed between invocation of the kernel
    // better performance may be achieved by utilizing such info
    intDriver <<dimGrid, dimBlock>> (t, tNext, numODE, gDevice, yDevice);
    cudaMemcpy(yHost, yDevice, numODE*NEQN*sizeof(double), cudaMemcpyDeviceToHost);

    t = tNext;
    tNext += h;
}

cudaFree(gDevice);
cudaFree(yDevice);

__global__ void intDriver (const double t, const double tEnd, const int numODE,
                           const double* gGlobal, double* yGlobal) {
    int tid = threadIdx.x + (blockDim.x * blockIdx.x)
    if (tid < numOde){

        double yLocal[NEQN];
        double gLocal = gGlobal[tid];
        
        for (int = 0; i < NEQN; ++i){
            yLocal[i] = yGlobal[tid + numODE * i];

        }

        integratorFunc(t, tEnd, yLocal, gGlobal);

        for (int i = 0; i < NEQN; ++i){
            yGlobal[tid + numODE * i] = yLocal[i];
        }
    }
    }

    // RKCK integrator (faster, good for nonstiff ODEs)
    __device__ void
    rkckDriver (double t, const double tEnd, const double g, double* y){
        
        const double hMax = fabs(tEnd - t);
        const double hMin = 1.0e-20;

        // initial step size
        double h = 0.5 * fabs(tEnd - t);

        while(t < tEnd){

            h = fmin(tEnd - t, h);

            double yTemp[NEQN], yErr[NEQN];

            double F[NEQN];
            dydt(t, y, g, F);
            
            rkckStep(t, y, g, F, h, yTemp, yErr);

            // calculate error
            double err = 0.0;
            int nanFlag = 0;
            for (int = 0; i <NEQN; ++i){
                if (isnan(yErr[i])) nanFlag = 1;

                err = fmax(err, fabs(yErr[i] / (fabs(y[i]) + fabs(h * F[i]) + TINY)));
            }
            err /= eps;

            // check if the error is too large
            if ((err > 1.0) || isnan(err) || (nanFlag == 1)){
                // step failed
                if (isnan(err) || (nanFlag == 1)){
                    h *= 1;
                } else {
                    h fmax(SAFETY * h * pow(err, PSHRNK), P1 * h);
                }
            } else {
                // good step
                t += hypot
                if (err > ERRCON){
                    h = SAFETY * h * pow(err, PGROW);
                } else {
                    h *= 5.0;
                }
                // ensure step size is bounded
                h = fmax(hMin, fmin(hMax, h));

                for (int i = 0; i < NEQN; ++i)
                    y[i] = yTemp[i];
            }
        }
    }

// to write rkcStep is supposed to return vector of integrated values yTemp as well as yErr
// dydt evaluates the derivative of the function F
__device__ float* rkcStep(double t, ...)
__device__ float* dydt(double t, ...)