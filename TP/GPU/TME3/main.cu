#include <stdio.h>
#include "timer.h"


void addVect(int *a, int *b, int *c, int length){

	int i;

	for(i=0; i<length; i++){
		c[i] = a[i] + b[i];
	}
}


__global__ void addVect_k(int *a, int *b, int *c, int length){

	int i = threadIdx.x + blockIdx.x*blockDim.x;

	/*while(i<length){
		c[i] = a[i] + b[i];
		i += blockDim.x*gridDim.x;
    }*/

	if(i<length){
		c[i] = a[i] + b[i];
    }
}



int main (void){

	// Variables definition
	int *a, *b, *c;
	int *aGPU, *bGPU, *cGPU;
	int i;
	
	// Length for the size of arrays
	int length = 1e8;

	Timer Tim;							// CPU timer instructions

	// Memory allocation of arrays 
	a = (int*)malloc(length*sizeof(int));
	b = (int*)malloc(length*sizeof(int));
	c = (int*)malloc(length*sizeof(int));

	cudaMalloc(&aGPU, length*sizeof(int));
	cudaMalloc(&bGPU, length*sizeof(int));
	cudaMalloc(&cGPU, length*sizeof(int));

	// Setting values
	for(i=0; i<length; i++){
		a[i] = i;
		b[i] = 9*i;
	}

	cudaMemcpy(aGPU, a, length*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(bGPU, b, length*sizeof(int), cudaMemcpyHostToDevice);


	Tim.start();						// CPU timer instructions

	// Executing the addition 
	//addVect(a, b, c, length);

	float TimeVar;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	//addVect_k<<<64, 256>>>(aGPU, bGPU, cGPU, length);


	addVect_k<<<(length+255)/256, 256>>>(aGPU, bGPU, cGPU, length);


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&TimeVar, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);



	Tim.add();							// CPU timer instructions



	cudaMemcpy(c, cGPU, length*sizeof(int), cudaMemcpyDeviceToHost);

	// Displaying the results to check the correctness 
	for(i=length-50; i<length-45; i++){
		printf(" ( %i ): %i\n", a[i]+b[i], c[i]);
	}

	//printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
	//	   (float)Tim.getsum());			// CPU timer instructions

	printf("GPU Timer for the addition on the GPU of vectors: %f s\n", 
		   TimeVar/1000.0f);			// CPU timer instructions


	// Freeing the memory
	cudaFree(aGPU);
	cudaFree(bGPU);
	cudaFree(cGPU);
	free(a);
	free(b);
	free(c);

	return 0;
}