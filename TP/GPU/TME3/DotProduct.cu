#include <stdio.h>
#include "timer.h"

#define NB 512
#define NTPB 1024

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// reduction use global memeory
__device__ float cGlob[NB * NTPB];
__global__ void GlobAtomic_kernel(float *A, float *B, float *C) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    cGlob[idx] = A[idx] * B[idx];
    atomicAdd(&(C[0]), cGlob[idx]);
}

// reduction use shared memory
__global__ void ShaGlobAtomic_kernel(float *A, float *B, float *C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    
    // Make multiplication in parallele and for each bloc, threads wait for each other
    __shared__ float SC[NTPB];
    SC[threadIdx.x] = A[idx] * B[idx];
    __syncthreads();

    i = blockDim.x / 2;
    while(i != 0 ) {
        if(threadIdx.x < i) {
            SC[threadIdx.x] += SC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(threadIdx.x == 0) {
        atomicAdd(&(C[0]), SC[0]);
    }
}


int main(void) {
    Timer Tim;	// CPU timer instructions

    float *d_a, *d_b, *d_c;	// device copies of a, b, c
    
    int length  = NB * NTPB;
    int size = sizeof(float);

    float *a = (float*)malloc(length * size);
    float *b = (float*)malloc(length * size);
    float *c = (float*)malloc(size);

    for(int i=0; i<length; i++)     {
        a[i] = 1;
        b[i] = 1;
	}
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, length * size);
    cudaMalloc((void **)&d_b, length * size);
    cudaMalloc((void **)&d_c, size);


    // printf("size: %d\n", size);
    // printf("length dc %ld\n", sizeof(*a));
    // printf("length dc %ld\n", sizeof(*d_c));
    // printf("length db %ld\n", sizeof(d_b));

    // Copy inputs to device
    cudaMemcpy(d_a, a, length * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, length * size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    Tim.start();						// CPU timer instructions
    ShaGlobAtomic_kernel<<<NB, NTPB>>>(d_a, d_b, d_c); // weird !!!!!!!!!!!!!!!!
    // GlobAtomic_kernel<<<NB, NTPB>>>(d_a, d_b, d_c);
    Tim.add();							// CPU timer instructions
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    printf("c = %f\n", *c);
    printf("CPU Timer for the The dot product: %f s\n", (float)Tim.getsum()); // CPU timer instructions
    return 0;
}