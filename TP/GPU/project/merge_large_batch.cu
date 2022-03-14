#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

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

bool sum_len_equal_d(int *A, int *B, int N, int d){
	for (int i = 0; i < N; i++){
		if (A[i] + B[i] != d){
			return false;
		}
	}
	return true;
}

bool power_of_two(int x){
	return (x != 0) && ((x & (x - 1)) == 0);
}

__global__ void mergeSmallBatch_k(int * A, int *B, int *C, int * sizeA, int *sizeB, const int d, const int N){
    const int tidx = threadIdx.x % d; // Diagonal number in number Qt table
    const int Qt = (threadIdx.x - tidx) / d; // batch table index in shared memory of a bloc
    const int gbx = Qt + blockIdx.x * (blockDim.x / d); // batch table index in shared memory glob memory

    __shared__ int shared_A[1024]; // shared memory for part of values in A
    __shared__ int shared_B[1024]; // shared memory for part of values in B

    const int sizeAi = sizeA[gbx]; // Ai batch size
    const int sizeBi = sizeB[gbx]; // Bi batch size

    shared_A[Qt * d + tidx]  = A[gbx * d + tidx];
    shared_B[Qt * d + tidx]  = B[gbx * d + tidx];

    __syncthreads(); // 

    if (gbx * d + tidx >= N * d){
		return;
	}

    int Kx, Ky, Px, Py, Qx, Qy, offset;

    /*run small path merge*/
    if(tidx > sizeAi){
		Kx = tidx - sizeAi;
		Ky = sizeAi;
		Px = sizeAi;
		Py = tidx - sizeAi;
	}else{
		Kx= 0;
		Ky= tidx;
		Px= tidx;
		Py= 0;
	}

	while(true){
		offset = (Ky-Py)/2;
		Qx = Kx + offset;
		Qy = Ky - offset;
		if(Qx <= sizeBi && (Qy == sizeAi || Qx == 0  || shared_A[Qt*d + Qy] > shared_B[Qt*d + Qx - 1])){
			if(Qx == sizeBi || Qy == 0 || shared_A[Qt*d + Qy - 1] <= shared_B[Qt*d + Qx]){
				if(Qy < sizeAi && (Qx == sizeBi || shared_A[Qt*d + Qy] <= shared_B[Qt*d + Qx])){
					C[gbx * d + tidx] = shared_A[Qt*d + Qy];
				}else{
					C[gbx * d + tidx] = shared_B[Qt*d + Qy];
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

void wrapper_batch_merge(int * A, int *B, int *CGPU, int * sizeA, int *sizeB, const int d, int N, int NTPB, int NB){
    /*CUDA memory allocation*/
	int * AGPU, * BGPU;
    int * sizeAGPU, * sizeBGPU;
    cudaMalloc(&AGPU, N*d*sizeof(int));
    cudaMalloc(&BGPU, N*d*sizeof(int));
    cudaMalloc(&sizeAGPU, N*sizeof(int));
    cudaMalloc(&sizeBGPU, N*sizeof(int));

    /*Copy arrays from CPU to GPU*/
    cudaMemcpy(AGPU, A, N*d * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(BGPU, B, N*d * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sizeAGPU, sizeA, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sizeBGPU, sizeB, N * sizeof(int), cudaMemcpyHostToDevice);

    /*Run GPU kernel to fusion AGPU and BGPU into CGPU*/
    mergeSmallBatch_k<<<NTPB, NB>>>(AGPU, BGPU, CGPU, sizeAGPU, sizeBGPU, d, N);
    cudaFree(AGPU);
    cudaFree(BGPU);
    cudaFree(sizeAGPU);
    cudaFree(sizeBGPU);
}

int main(void){
    srand(0); //random seed
    /*Parameters initialisation*/
    const int d = 32;
    const int N = 10000;
    const int NTPB = 1024;
    const int NB = (NTPB - 1 + N*d)/NTPB;

    /*CPU memory allocation*/
    int *A = (int*)malloc(N*d * sizeof(int));
    int *B = (int*)malloc(N*d * sizeof(int));
    int *C = (int*)malloc(N*d * sizeof(int));
	int * size_batchs_A = (int*) malloc(N * sizeof(int));
	int * size_batchs_B = (int*) malloc(N * sizeof(int));
    /*GPU memory allocation for C(merge array)*/
    int *CGPU;
    cudaMalloc(&CGPU, N*d*sizeof(int));

    /*Array initialisation*/
    for(int i = 0; i < N; i++){
        int rand_size = rand() % d;
        size_batchs_A[i] = rand_size;
        size_batchs_B[i] = d - size_batchs_A[i];

        for(int j = 0; j < size_batchs_A[i]; j++){
            A[i*d + j] = j;
        }
        for(int j = 0; j < size_batchs_B[i]; j++){
            B[i*d + j] = j + 1;
        }
    }
    assert(power_of_two(d));
    assert(power_of_two(NTPB));
    assert(NTPB <= 1024);
    assert(NTPB % d == 0);
    assert(sum_len_equal_d(size_batchs_A, size_batchs_B, N,d));

    wrapper_batch_merge(A, B, CGPU, size_batchs_A, size_batchs_B, d, N, NTPB, NB);

    // Show merging result
    int i = N - 1;

    for(int j = 0; j < d; j++){
        printf("|%d| ", C[i*d + j]);
    }
    printf("\n");

    free(A);
    free(B);
    free(C);
    free(size_batchs_A);
    free(size_batchs_B);

    cudaFree(CGPU);
    return 0;
}