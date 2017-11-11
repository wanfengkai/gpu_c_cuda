#include <stdio.h>
#define TPB 256
#define B 1

__global__ void hello()
{
 printf("Hello World ! My thread id is %2d \n",threadIdx.x);
}

int main()
{
 hello<<<B,TPB>>>();
 cudaDeviceSynchronize();
 return 0;
}
