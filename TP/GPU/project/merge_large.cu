#include <stdio.h>
// #include <stdlib.h>
#include <cuda/std/utility>
#include "timer.h"

#define NB 1
#define NTPB 1024

#define len_A 512
#define len_B 512

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

__global__ void kernel_merge_small(int *a, int *b, int *M, int lengthA, int lengthB){
	// A represents the biggest array
	int *A, *B, la, lb, offset;
	if(lengthA > lengthB ){ 
        A = a;
		B = b;
		la = lengthA;
		lb = lengthB;
	}else{
		B = a;
		A = b;
		lb = lengthA;
		la = lengthB;
	}

	int i = threadIdx.x;
	int Kx, Ky, Px, Py, Qx, Qy;

	if(i >= lengthA + lengthB){
		return; 
	}
	if(i> la){
		Kx= i-la;
		Ky= la;
		Px= la;
		Py= i-la;
	
	}
	else{
		Kx= 0;
		Ky= i;
		Px= i;
		Py= 0;

	}
	while(true){
		offset = (Ky-Py)/2;
		Qx = Kx + offset;
		Qy = Ky - offset;
		if(Qx<= lb && (  Qy==la || Qx==0  || A[Qy]>B[Qx -1] )){
			if(Qx == lb || Qy == 0 || A[Qy-1]<=B[Qx] ){
				if(Qy< la && (Qx == lb || A[Qy] <= B[Qx])){
					M[i]= A[Qy];
				}else{
					M[i]= B[Qx];
				}
				break;
			}else{
				Kx= Qx +1;
				Ky = Qy -1;		
			}
		}else{
			Px= Qx -1;
			Py = Qy +1;		
		}
	}

}



void wrapper_merge_large(int *A, int *B, int *C) {
    int *aGPU, *bGPU, *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&aGPU, len_A*sizeof(int)));
    testCUDA(cudaMalloc(&bGPU, len_B*sizeof(int)));
    testCUDA(cudaMalloc(&cGPU, (len_A + len_B)*sizeof(int)));

    testCUDA(cudaMemcpy(aGPU, A, len_A*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(bGPU, B, len_B*sizeof(int), cudaMemcpyHostToDevice));
    
    //start of kernel
    kernel_merge_small<<<NB, NTPB>>>(aGPU, bGPU, cGPU, len_A, len_B);

    testCUDA(cudaMemcpy(C, cGPU, (len_A + len_B)*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    printf("Execution time: %f ms\n", TimerV);

    testCUDA(cudaFree(aGPU));
    testCUDA(cudaFree(bGPU));
    testCUDA(cudaFree(cGPU));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
}

int main(void){
    int *A = (int*)malloc(len_A * sizeof(int));
    int *B = (int*)malloc(len_B * sizeof(int));
    int *C = (int*)malloc((len_A + len_B) * sizeof(int));
    for(int i=0; i<len_A; i++){
        A[i] = i;
	}
    for(int i=0; i<len_B; i++){
        B[i] = i;
    }
    wrapper_merge_large(A, B, C);
    for(int i = 0; i < len_A + len_B; i++){
        printf("%d ", C[i]);
    }

    return 0;
}