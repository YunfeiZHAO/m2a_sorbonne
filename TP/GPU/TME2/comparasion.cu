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

__device__ int aGlob[NB * NTPB];
__device__ int bGlob[NB * NTPB];

// __constant__ int aGlob[NB * NTPB];
// __constant__ int bGlob[NB * NTPB];

// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

//**** perform a large number of global memory access
__global__ void kernel_vect(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    for(i = 0; i < 20; i++) {
        aGlob[idx] = a[idx] + 1;
        bGlob[idx] = b[idx] + 1;
        a[idx] = aGlob[idx];
        b[idx] = bGlob[idx];
    }
    c[idx] = aGlob[idx] + bGlob[idx];
}

//**** replace globle memory by the register memory access
__global__ void kernel_vect_register(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i;

    int aLocal1, bLocal1, aLocal2, bLocal2;
    aLocal1 = a[idx] + 1;
    bLocal1 = b[idx] + 1;
    for(i = 0; i < 19; i++) {
        aLocal2 = aLocal1 + 1;
        bLocal2 = bLocal1 + 1;
        aLocal1 = aLocal2;
        bLocal1 = bLocal2;
    }
    c[idx] = aLocal1 + bLocal1;
}

//**** replace globle memory by shared memory access
__global__ void kernel_vect_shared(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i;

    int idxShared = threadIdx.x;
    __shared__ int sA1[NTPB], sA2[NTPB];
    __shared__ int sB1[NTPB], sB2[NTPB];
    sA1[idxShared] = a[idx] + 1;
    sB1[idxShared] = b[idx] + 1;
    for(i = 0; i < 19; i++) {
        sA2[idxShared] = sA1[idxShared] + 1;
        sB2[idxShared] = sB1[idxShared] + 1;
        sA1[idxShared] = sA2[idxShared];
        sB1[idxShared] = sB2[idxShared];
    }
    c[idx] = sA1[idxShared] + sB1[idxShared];
}

void wrapper_vect(int *a, int *b, int *c) {
    int *aGPU, *bGPU, *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&aGPU, NB*NTPB*sizeof(int)));
    testCUDA(cudaMalloc(&bGPU, NB*NTPB*sizeof(int)));
    testCUDA(cudaMalloc(&cGPU, NB*NTPB*sizeof(int)));

    testCUDA(cudaMemcpy(aGPU, a, NB*NTPB*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(bGPU, b, NB*NTPB*sizeof(int), cudaMemcpyHostToDevice));

    // Comparaison
    kernel_vect<<<NB, NTPB>>>(aGPU, bGPU, cGPU);
    // kernel_vect_register<<<NB, NTPB>>>(aGPU, bGPU, cGPU);
    // kernel_vect_shared<<<NB, NTPB>>>(aGPU, bGPU, cGPU);

    testCUDA(cudaMemcpy(c, cGPU, NB*NTPB*sizeof(int), cudaMemcpyDeviceToHost));
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

int main (void){
    int length  = NB * NTPB;
    int *a = (int*)malloc(length * sizeof(int));
    int *b = (int*)malloc(length * sizeof(int));
    int *c = (int*)malloc(length * sizeof(int));
    for(int i=0; i<length; i++){
        a[i] = 1;
        b[i] = 1;
	}
    wrapper_vect(a, b, c);
    // for(int i=length-50; i<length-45; i++){
	// 	printf(" ( %i ): %i\n", a[i] + 20 +b[i] + 20, c[i]);
	// }
    return 0;
}