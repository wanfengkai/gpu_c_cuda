#include <stdio.h>
#include <sys/time.h>
__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
//  printf("Computing SAXPY on the GPU Done!");
}

int main(void)
{

  
  int N = 1<<20; 
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  float maxError= 0.0f;
  // perform cpu saxpy
 float *c_y;
 struct timeval tv;
suseconds_t time_end_cpu;
suseconds_t time_start_cpu;
suseconds_t time_cpu;
float A = 2.0f;
int success1=gettimeofday(&tv, NULL);
if(!success1){

  time_start_cpu = tv.tv_usec;
   printf("time start %06ld \n",time_start_cpu);
}
// c_x = (float*)malloc(N*sizeof(float));
  c_y = (float*)malloc(N*sizeof(float));

 for (int i= 0;i<N;i++){
 c_y[i]=A*x[i]+ y[i];

}
 printf("Computing SAXPY on the CPU Done!\n");
int success2=gettimeofday(&tv, NULL);

if(!success2){
 
 time_end_cpu = tv.tv_usec;
 printf("time end %06ld \n",time_end_cpu);
  time_cpu = time_end_cpu - time_start_cpu;
}

printf("time needed for cpu: %06ld  \n",time_cpu);

 for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(c_y[i]-4.0f));
  printf("Max error for CPU is: %f\n", maxError);



  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
 printf("Computing SAXPY on the GPU Done!\n");

//  float maxError = 0.0f;
  maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error for GPU is: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
