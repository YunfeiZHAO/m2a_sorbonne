/**************************************************************
This code is a part of a course on cuda taught by the author: 
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#include <stdio.h>

#define NB 16384
#define NTPB 512

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}


// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))



__device__ void Test(int *a) {

	for (int i = 0; i < 1000; i++) {
		*a = *a + 1;
	}
}

__device__ int aGlob[NB*NTPB];					// Global variable solution

__global__ void MemComp(int *a){

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	aGlob[idx] = a[idx];

	Test(aGlob + idx);

	a[idx] = aGlob[idx];
}


int main (void){

	
	int *a, *aGPU;
	float Tim;										// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
		
	a = (int*)malloc(NB*NTPB*sizeof(int));
	testCUDA(cudaMalloc(&aGPU, NB*NTPB * sizeof(int)));

	for(int i=0; i<NB; i++){
		for(int j=0; j<NTPB; j++){
			a[j+i*NTPB] = j+i*NTPB;
		}
	}

	testCUDA(cudaMemcpy(aGPU, a, NB*NTPB*sizeof(int), cudaMemcpyHostToDevice));

	testCUDA(cudaEventRecord(start, 0));			// GPU timer instructions
	
	for(int i = 0; i<100; i++) {
		MemComp<<<NB,NTPB>>>(aGPU);
	}
	
	testCUDA(cudaEventRecord(stop, 0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&Tim,				// GPU timer instructions
		start, stop));								// GPU timer instructions

	printf("Time per execution: %f ms\n", Tim/100);
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	testCUDA(cudaMemcpy(a, aGPU, NB*NTPB*sizeof(int), cudaMemcpyDeviceToHost));
	testCUDA(cudaFree(aGPU));

	for(int i= 0; i<4; i++){
		printf("%i = %i \n", 100000 + i, a[i]);
	}

	free(a);

	return 0;
}
