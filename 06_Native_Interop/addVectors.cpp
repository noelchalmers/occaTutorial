#include <hip/hip_runtime.h>

// Native HIP addVectors kernel
extern "C" 
__global__ __launch_bounds__(256) 
void addVectors(const int entries,
                const float * __restrict__ a,
                const float * __restrict__ b,
                      float * __restrict__ ab) {

  const int n = threadIdx.x + blockDim.x * blockIdx.x;

  if (n < entries) {
    ab[n] = a[n] + b[n];
  }
}
