#include <ctime>
#include <stdio.h>
#include <math.h>
#include <limits.h>

#define NB 1
#define NTPB 1024

#define lenC 50


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


__global__ void merge(int *a, int *b, int *c, int d, int len_new){
	// A represents the biggest array

    __shared__ int A_copy[lenC];
    __shared__ int B_copy[lenC];
    __shared__ int C_copy[lenC];

    for(int i=0;i<lenC;i++){
        C_copy[i] = c[i];
    }

	int *A, *B, *C, la, lb, offset;
    int Kx, Ky, Px, Py, Qx, Qy;

    A = a;
    B = b;
    C = c;

    int block = 0;
    int i = threadIdx.x;
    int i_d = i%d;
    int block_d = i/d;

    int start_x, start_y, end_x, end_y;

    if(i>=len_new)
        return;

    start_x = d*block_d;
    start_y = d*(block_d+1);

    end_x = d*(block_d+1);
    end_y = d*(block_d+2);

    if(block_d%2 == 0)
        A[i] = c[i];
    else
        B[i-d] = c[i];
}

void wrapper_partition(int *A, int *B, int *C, int p, int len_new){
    int *aGPU, *bGPU, *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&aGPU, len_new*sizeof(int)));
    testCUDA(cudaMalloc(&bGPU, len_new*sizeof(int)));
    testCUDA(cudaMalloc(&cGPU, len_new*sizeof(int)));   
    testCUDA(cudaMemcpy(cGPU, C, len_new*sizeof(int), cudaMemcpyHostToDevice));

    //start of kernel
    // partition<<<NB, 1>>>(aGPU, bGPU, a_partGPU, b_partGPU, len_A, len_B);
    
    // for(int l = 0; l<=8; l++)
    merge<<<1, NTPB>>>(aGPU, bGPU, cGPU, 8, len_new);

    testCUDA(cudaMemcpy(C, cGPU, len_new*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(A, aGPU, len_new*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(B, bGPU, len_new*sizeof(int), cudaMemcpyDeviceToHost));

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
    // On veut trier C!!
    // on va copier sur A et B des morceaux de C 
    int p = (int) floor(log2(lenC));
    int to_add = ((int) pow(2, p+1))%lenC;

    int len_new = lenC+ to_add;

    int *A = (int*)malloc(len_new* sizeof(int));
    int *B = (int*)malloc(len_new* sizeof(int));
    int *C = (int*)malloc(len_new* sizeof(int));

    for(int i=0; i<lenC; i++){
        C[i] = rand() %250;
	}
    for(int i=lenC; i<lenC + to_add; i++){
        C[i] = INT_MAX;
	}

    wrapper_partition(A, B, C, p, len_new);

    printf("\n");
    for(int i = 0; i < len_new ; i++){
        printf("| %d |", C[i], i);
    }
    printf("\n\n");
    for(int i = 0; i < len_new ; i++){
        printf("| %d |", A[i], i);
    }
    printf("\n\n");
    for(int i = 0; i < len_new ; i++){
        printf("| %d |", B[i], i);
    }

    return 0;
}