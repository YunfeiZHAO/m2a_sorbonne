 #include <ctime>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <cuda.h>

#define NTPB 1000
#define lenC 2000

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


__global__ void merge(int *C, int d){

    __shared__ int C_sh[NTPB];

	int la, lb, offset;
    int Kx, Ky, Px, Py, Qx, Qy;

    int i = threadIdx.x;
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x -tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);
    
    if((i+ blockIdx.x*blockDim.x)>=lenC)
        return;

    C_sh[i] = C[i+ blockIdx.x*blockDim.x];
    __syncthreads();

	int longeur= blockIdx.x == (gridDim.x -1 ) ? (lenC - blockIdx.x*blockDim.x) : blockDim.x;
	int reste= longeur-(longeur/d)*d;
	if((longeur/d)*d > i){
		la = d/2;
		lb = d/2;
	}else if(reste <= d/2){
		return;
	}else{
		la= d/2;
		lb= reste - d/2;
	}
    
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

    int move_a = d*Qt;
    int move_b = d*Qt + d/2;

	while(true){
		offset = (Ky-Py)/2;
		Qx = Kx + offset;
		Qy = Ky - offset;
		if( Qx<= lb && (  Qy==la || Qx==0  || C_sh[Qy+move_a]>C_sh[Qx-1+move_b])){
			if(Qx == lb || Qy == 0 || C_sh[Qy-1+ move_a]<=C_sh[Qx+ move_b] ){
				if(Qy< la && (Qx == lb || C_sh[Qy+move_a] <= C_sh[Qx+move_b])){
          C[i+ blockIdx.x*blockDim.x]= C_sh[Qy+move_a];
        }else{
          C[i+ blockIdx.x*blockDim.x]= C_sh[Qx+move_b]; 
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

 void wrapper_partition(int *C, int NB){
    int *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&cGPU, lenC*sizeof(int)));   
    testCUDA(cudaMemcpy(cGPU, C, lenC*sizeof(int), cudaMemcpyHostToDevice));

    for(int l = 1; l<= (int) ceil ( log2(NTPB)); l++){ 
        merge<<<NB, NTPB >>>(cGPU, pow(2,l));
    }

    testCUDA(cudaMemcpy(C, cGPU, lenC*sizeof(int), cudaMemcpyDeviceToHost));

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    printf("Execution time: %f ms\n", TimerV);

    testCUDA(cudaFree(cGPU));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
}

int main(void){

    int NB = (int) ceil(lenC/(float)NTPB);

    printf("NB: %d\n", NB);

    int *C = (int*) malloc(lenC* sizeof(int));
    for(int i=0; i<lenC; i++){
        C[i] = 3000-i;
	  }    
    
    wrapper_partition(C, NB);

    printf("\n");
    for(int i = 0; i < lenC ; i++){
        printf("| %d ", C[i]);
    }
    printf("\n");

    return 0;
}