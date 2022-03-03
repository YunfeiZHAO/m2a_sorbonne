#include <stdio.h>
#include "timer.h"

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main(void) {
    Timer Tim;	// CPU timer instructions
    int a, b, c;	// host copies of a, b, c
    int *d_a, *d_b, *d_c;	// device copies of a, b, c
    int size = sizeof(int);
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    //d_c = (int*)malloc(sizeof(int));
    cudaMalloc((void **)&d_c, size);
    // Setup input values
    a = 100;
    b = 10000;
    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    Tim.start();						// CPU timer instructions
    add<<<1,1>>>(d_a, d_b, d_c);
    Tim.add();							// CPU timer instructions
    // Copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    printf("c=%d\n", c);
    printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
    (float)Tim.getsum());			// CPU timer instructions
    return 0;
}