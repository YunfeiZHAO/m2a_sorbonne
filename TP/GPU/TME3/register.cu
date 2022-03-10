#include <stdio.h>
#include "timer.h"

#define NB 512
#define NTPB 1024

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line){
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void reg_kernel(int *A){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int lane = threadIdx.x % warpSize;
	int loc;
	loc = A[idx];
	if(blockIdx.x < 1 && lane == idx) {
		printf("%d, %d, %d\n", lane, loc, __shfl_down_sync(loc, 1, warpSize));
	}
	printf("%d, %d, %d\n", lane, loc, __shfl_down_sync(loc, 1, warpSize));
}

int main(void){
	int length = NB*NTPB;
	int *aGPU;
	int *a = (int*)malloc(length * sizeof(int));
	for(int i=0; i<length; i++){
        a[i] = 1;
	}
	cudaMalloc(&aGPU,length*sizeof(int));
	cudaMemcpy(aGPU, a, length*sizeof(int), cudaMemcpyHostToDevice);
	reg_kernel<<<NB, NTPB>>>(aGPU);
    // testCUDA(cudaMemcpy(c, cGPU, NB*NTPB*sizeof(int), cudaMemcpyDeviceToHost));
	return 0;
}

// This wont show and how does shuffle works