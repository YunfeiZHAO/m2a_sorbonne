#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define N 2048

__global__ void block_add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void thread_add(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x]+ b[threadIdx.x];
}

void random_ints(int* x, int size)
{
	int i;
	for (i=0;i<size;i++) {
		x[i]=rand()%10;
	}
}

int main(void) {
    Timer Tim;							// CPU timer instructions
    int *a, *b, *c;	// host copies of a, b, c
    int *d_a, *d_b, *d_c;	// device copies of a, b, c
    int size = N * sizeof(int);
    // Alloc space for devicecopies of a, b, c
    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);
    // Allocspace for host copies of a, b, c and setup input values
    a = (int*)malloc(size); 
    random_ints(a, N);

    b = (int*)malloc(size); 
    random_ints(b, N);

    c = (int*)malloc(size);
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with N blocks

    Tim.start();						// CPU timer instructions
    // block_add<<<N,1>>>(d_a, d_b, d_c);
    thread_add<<<1,N>>>(d_a, d_b, d_c);
    Tim.add();							// CPU timer instructions

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    //show
    // for (int i=0;i<N;i++) {
	// 	printf("a[%d]=%d , b[%d]=%d, c[%d]=%d\n",i,a[i],i,b[i],i,c[i]);
	// }
    printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
        (float)Tim.getsum());			// CPU timer instructions
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}

