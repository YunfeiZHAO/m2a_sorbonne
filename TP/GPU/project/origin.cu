#include <ctime>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <cuda.h>

#define NTPB 1024
#define lenC 1025

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


__global__ void merge(int *C, int d, int len_new){

    __shared__ int C_sh[NTPB];

	int la, lb, offset;
    int Kx, Ky, Px, Py, Qx, Qy;

    int i = threadIdx.x;
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x -tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);
    
    if(i>=len_new)
        return;

    C_sh[Qt*d + tidx] = C[gbx*d + tidx];
    __syncthreads();

    la = d/2;
    lb = d/2;
    
    if(tidx > la){
		Kx= tidx-la;
		Ky= la;
		Px= la;
		Py= tidx-la;
	}
	else{
		Kx= 0;
		Ky= tidx;
		Px= tidx;
		Py= 0;
    }

    int move_a = (d*Qt)%1024;
    int move_b = (d*Qt + d/2)%1024;

    int move_batch_global = d*gbx;
	while(true){
		offset = (Ky-Py)/2;
		Qx = Kx + offset;
		Qy = Ky - offset;
		if(Qy >= 0 && Qx<= lb && (  Qy==la || Qx==0  || C_sh[Qy+move_a]>C_sh[Qx-1+move_b])){
			if(Qx == lb || Qy == 0 || C_sh[Qy-1+ move_a]<=C_sh[Qx+ move_b] ){
				if(Qy< la && (Qx == lb || C_sh[Qy+move_a] <= C_sh[Qx+move_b])){
					C[tidx + move_batch_global]= C_sh[Qy+move_a];
				}else{
					C[tidx + move_batch_global]= C_sh[Qx+move_b];
				}
				break;
			}else{
				Kx= Qx+1;
				Ky = Qy-1;		
			}
		}else{
			Px= Qx-1;
			Py = Qy+1;		
		}
	}  
}

void wrapper_partition(int *C, int p, int len_new, int NB){
    int *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&cGPU, len_new*sizeof(int)));   
    testCUDA(cudaMemcpy(cGPU, C, len_new*sizeof(int), cudaMemcpyHostToDevice));

    for(int l = 1; l<=p; l++){ 
        merge<<<NB, NTPB >>>(cGPU, pow(2,l), len_new);
    }

    testCUDA(cudaMemcpy(C, cGPU, len_new*sizeof(int), cudaMemcpyDeviceToHost));

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    printf("Execution time: %f ms\n", TimerV);

    testCUDA(cudaFree(cGPU));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
}

int main(void){

    int p = (int) ceil(log2(lenC));
    int to_add = ((int) pow(2, p))%lenC;
    int len_new = lenC + to_add;
    int NB = len_new/NTPB;

    printf("len_new: %d\n", len_new);
    printf("NB: %d\n", NB);

    int *C = (int*) malloc(len_new* sizeof(int));

    for(int i=0; i<lenC; i++){
        C[i] = 1500-i;
	}

    for(int i=lenC; i<len_new; i++){
        C[i] = INT_MAX;
	}
    printf("P: %d\n", p);
    
    wrapper_partition(C, p, len_new, NB);

    printf("\n");
    for(int i = 0; i < len_new ; i++){
        printf("| %d |", C[i]);
    }
    printf("\n");

    return 0;
}