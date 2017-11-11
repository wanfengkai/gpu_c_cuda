# include <stdio.h>
# include <cuda_runtime.h>
# include <sys/time.h>
# include <string.h>
# define CHECK(call)                                                    \
{									\
	const cudaError_t error = call;					\
	if (error != cudaSuccess)					\
	{								\
		printf("Error: %s: %d \n",__FILE__,__LINE__);		\
		printf("reason: %s \n",cudaGetErrorString(error));	\
		exit(1);						\
	}								\
}
// number of iteration, number pf paticle elements									
# define ITER  10
//# define N  5000
// define struct of Particles, don't know if float3 type can be used in cpu.
struct Particle
{
	float3 position;
	float3 velocity;
};


double cpuSecond(){
// this returns the time in double format.
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec+(double)tp.tv_usec*1.e-6);

}

void checkResult(struct Particle *hostRef,struct Particle *gpuRef,const int size){
// Check if the cpu and gpu implementation is the same.
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i=0; i<size ; i++){
// check on x axis
		if(abs(hostRef[i].position.x-gpuRef[i].position.x)>epsilon){
			match =0;
			printf("array not match on x axis \n");
			printf("host %5.2f gpu %5.2f at current %d \n",hostRef[i].position.x,gpuRef[i].position.x, i);
		        break;
		}
		
// check on y axis
                if(abs(hostRef[i].position.y-gpuRef[i].position.y)>epsilon){
                        match =0;
                        printf("array not match on y axis \n");
                        printf("host %5.2f gpu %5.2f at current %d \n",hostRef[i].position.y,gpuRef[i].position.y, i);	
			break;
                }
// check on z axis

                if(abs(hostRef[i].position.z-gpuRef[i].position.z)>epsilon){
                        match =0;
                        printf("array not match on z axis \n");
                        printf("host %5.2f gpu %5.2f at current %d \n",hostRef[i].position.z,gpuRef[i].position.z, i);
			break;
                }
}

	if (match) printf("Match!\n");
}

void randomacc(float3 *acc, const int size)
{
// this generate randome accelaration with the seed of time
	time_t t;
	srand((unsigned) time(&t));
	
	for (int i=0; i < size; i++)
	{ 
	// generate accelerate
		acc[i].x = (float) (rand() &0xFF) / 10.0f;
		acc[i].y = (float) (rand() &0xFE) / 10.0f;
		acc[i].z = (float) (rand() &0xFD) / 10.0f;
	}
}



void update_CPU(struct Particle *array,float3 *acc,int size)
{
// this is used to update the position of Particles wrt random acceleration for one iteration.
	for (int i=0;i<size;i++){
		// velocity update
		array[i].velocity.x += acc[i].x;
		array[i].velocity.y += acc[i].y;
		array[i].velocity.z += acc[i].z;
		// position update
		array[i].position.x += array[i].velocity.x;
		array[i].position.y += array[i].velocity.y;
		array[i].position.z += array[i].velocity.z;
	}


}



__global__ void update_GPU(struct Particle *array,float3 *acc,int size)
{
// this is used to update the position of Paticles wrt to random acc in GPU.
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<size)
	{
                // velocity update
                array[i].velocity.x += acc[i].x;
                array[i].velocity.y += acc[i].y;
                array[i].velocity.z += acc[i].z;
                // position update
                array[i].position.x += array[i].velocity.x;
                array[i].position.y += array[i].velocity.y;
                array[i].position.z += array[i].velocity.z;
	}
}


int main(){

	// malloc memory
	
	int block_size = 8;
	int rate =2;
	int  element_number = 20000;
	int block_size_iter = block_size;
	// create csv file to record.
	FILE *fp;
	printf("In this test we are going to test out with %ld iteration with fixing blokc_size of %d ", ITER, block_size );
	char *filename=(char *)"fixing_element_number.csv";
	fp=fopen(filename,"w+");
	fprintf(fp,"blokc_size,gpu_time_gap,cpu_time_gap \n");
	printf("created the csv file.\n");
	for (int i=0;i<ITER;i++)
   {    
        //element_number = rate * element_number;
	// for check out the blokc_size
//	element_number = element_number;
	size_t nBytes = element_number * sizeof(struct Particle);
	block_size_iter = block_size_iter*rate;

	struct Particle *particles,*hostRef,*gpuRef,*d_p;

	particles = (struct Particle *)malloc(nBytes);
	hostRef   = (struct Particle *)malloc(nBytes); 
	gpuRef    = (struct Particle *)malloc(nBytes);



	CHECK(cudaMalloc((struct Particle**)&d_p,nBytes));
	// transfer data to GPU memory
	CHECK(cudaMemcpy(d_p,particles,nBytes,cudaMemcpyHostToDevice));
	
	// get the random acc for cpu and gpu
	float3 *acc,*d_acc;
	// malloc memory to acc
	size_t acc_bytes= element_number* sizeof(float3);
	acc   = (float3 *)malloc(acc_bytes);
	randomacc(acc,element_number);
        CHECK(cudaMalloc((float3**)&d_acc,acc_bytes));
	// calculate the time needed for GPU start here.
	double d_start, d_gap;
	d_start = cpuSecond();
	CHECK(cudaMemcpy(d_acc,acc,acc_bytes,cudaMemcpyHostToDevice));
	update_GPU<<<((element_number+block_size_iter-1)/block_size_iter), block_size_iter>>>(d_p,d_acc,element_number);
	CHECK(cudaDeviceSynchronize());
	// copy the kernel result to host side
	cudaMemcpy(gpuRef,d_p,nBytes,cudaMemcpyDeviceToHost);
	// GPU timing end here.
	d_gap = cpuSecond()-d_start;
	printf("Summary: with block_size %d, particle number %d the time gap on GPU for one iteration is %f \n",block_size_iter,element_number,d_gap);
//	printf("gpuRef after copy %5.2f %5.2f\n",gpuRef[0].position.x,gpuRef[6].position.x);


	//update in cpu	
	// time count of cpu start here.

	double h_start,h_gap;
	h_start= cpuSecond();
	update_CPU(hostRef,acc,element_number);
//	printf("hostRef after operate  %5.2f  %5.2f\n",hostRef[0].position.x,hostRef[6].position.x);
	h_gap=cpuSecond()-h_start;
	printf("Summary:with the particle number of %d, the time gap on CPU for one iteration is  %f  \n ",element_number,h_gap);
	// compare the result if they match.
	checkResult(hostRef,gpuRef,element_number);		
      	// write line of data to record.
//	fprintf(fp,"%d,%f,%f \n",element_number,d_gap,h_gap);
	// write data  of block size 
	fprintf(fp,"%d,%f,%f \n",block_size_iter,d_gap,h_gap);
	// free the device global memory
	
	cudaFree(d_p);
	cudaFree(d_acc);
	// free the host global memory
	free(acc);
	free(particles);
	free(hostRef);
	free(gpuRef);
	

    } // end of for loop here.
	fclose(fp);
	return 0;

}
