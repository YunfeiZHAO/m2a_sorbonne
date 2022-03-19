#include <ctime>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <cuda.h>

#define NTPB 1024
#define lenC 4096

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

/**
 * For merge with d <= NTPB
 * */
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

    int move_a = (d*Qt)%NTPB;
    int move_b = (d*Qt + d/2)%NTPB;

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

/**
 * For patitioning merge with d > NTPB
 * */
__global__ void partition(int *C, int *A_part, int *B_part, int N_batch, int d, int NBPB){

    // *** Ne pas oublier le cas len_A == leb_B == 1 
    // *** Ne pas oublier NB >= min(len_A, len_B) 

	int Kx, Ky, Px, Py, Qx, Qy, offset;
    int i_batch = blockIdx.x/NBPB;
    int i_bloc_in_batch = blockIdx.x % NBPB;
    int i = i_bloc_in_batch * NTPB; //index of partition thread in batch
    int la = d/2;
    int lb = d/2;
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
    int move_a = i_batch * d; //to current batch 
    int move_b = i_batch * d + d/2;

	while(true){
		offset = (Ky-Py)/2;
		Qx = Kx + offset;
		Qy = Ky - offset;
		if(Qx<= lb && (  Qy==la || Qx==0  || C[Qy + move_a]>C[Qx -1 + move_b] )){
			if(Qx == lb || Qy == 0 || C[Qy-1 + move_a]<=C[Qx + move_b] ){
				A_part[blockIdx.x] = Qx + move_a;
                B_part[blockIdx.x] = Qy + move_b;
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

__global__ void merge_by_partition(int *C, int *A_part, int *B_part, int d, int NBPB, int NB){
    int Kx, Ky, Px, Py, Qx, Qy, offset;
    int start_x, start_y, end_x, end_y;
    // int i_batch = blockIdx.x/NBPB;
    // int i_bloc_in_batch = blockIdx.x % NBPB;

    start_x = A_part[blockIdx.x];
    start_y = B_part[blockIdx.x];

    if(blockIdx.x < NB-1){
        end_x = A_part[blockIdx.x + 1];
        end_y = B_part[blockIdx.x + 1];
    }else{
        end_x = NTPB * NB - d/2 -1;
        end_y = NTPB * NB - 1;
    }

    int lengthA = abs(end_x - start_x);
    int lengthB = abs(end_y - start_y);

    __shared__ int A_bloc_sh[NTPB];
    __shared__ int B_bloc_sh[NTPB];
    for(int l = 0; l< lengthA; l++){
        A_bloc_sh[l] = C[start_x + l];}
    for(int l = 0; l< lengthB; l++){
        B_bloc_sh[l] = C[start_y + l];}
    int i = threadIdx.x;
    if(i >= lengthA + lengthB){
		return; 
	}
    __syncthreads();
    int *A, *B;
    int la, lb;
	if(lengthA > lengthB){
        A = A_bloc_sh;
	    B = B_bloc_sh;
	    la = lengthA;
	    lb = lengthB;

	}else{
    	B = A_bloc_sh;
		A = B_bloc_sh;
		lb = lengthA;
		la = lengthB;
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
		if(Qy >= 0 && Qx<= lb && (  Qy==la || Qx==0  || A[Qy]>B[Qx -1] )){
			if(Qx == lb || Qy == 0 || A[Qy-1]<=B[Qx] ){
				if(Qy< la && (Qx == lb || A[Qy] <= B[Qx])){
					C[i + blockIdx.x * NTPB]= A[Qy];
				}else{
					C[i + blockIdx.x * NTPB]= B[Qx];
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

void wrapper_partitioning_merge(int *C, int d, int len_new, int NB){
    int *a_partGPU, *b_partGPU;
    int NBPB = d/NTPB; //number of bloc per batch
    int N_batch = len_new / d; //number of batch

    int *A_part = (int*)malloc(NBPB * N_batch * sizeof(int));
    int *B_part = (int*)malloc(NBPB * N_batch * sizeof(int));

    cudaMalloc(&a_partGPU, NBPB * N_batch * sizeof(int));
    cudaMalloc(&b_partGPU, NBPB * N_batch * sizeof(int));
    partition<<<NBPB * N_batch, 1>>>(C, a_partGPU, b_partGPU, N_batch, d, NBPB);
    testCUDA(cudaMemcpy(A_part, a_partGPU, NBPB * N_batch*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(B_part, b_partGPU, NBPB * N_batch*sizeof(int), cudaMemcpyDeviceToHost));

    printf("partition A\n");
    for(int i = 0; i < NBPB * N_batch; i++){
        printf("| %d |", A_part[i]);
    }
    printf("\n\n");
    printf("partition B\n");
    for(int i = 0; i < NBPB * N_batch ; i++){
        printf("| %d |", B_part[i]);
    }
    printf("\n\n");

    merge_by_partition<<<NB, NTPB>>>(C, a_partGPU, b_partGPU, d, NBPB, NB);
}

void wrapper_merge_order(int *C, int p, int len_new, int NB){
    int *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&cGPU, len_new*sizeof(int)));   
    testCUDA(cudaMemcpy(cGPU, C, len_new*sizeof(int), cudaMemcpyHostToDevice));

    for(int l = 1; l<=p; l++){ 
        int d = pow(2,l);
        if(d <= NTPB){
            merge<<<NB, NTPB >>>(cGPU, d, len_new);
        }else{
            wrapper_partitioning_merge(cGPU, d, len_new, NB);
        }
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
        //C[i] = 2048-i;
        C[i] = rand() % 100000;
	}

    for(int i=lenC; i<len_new; i++){
        C[i] = INT_MAX;
	}
    printf("P: %d\n", p);
    
    wrapper_merge_order(C, p, len_new, NB);

    printf("\n");
    for(int i = 0; i < len_new ; i++){
        printf("| %d |", C[i]);
    }
    printf("\n");

    return 0;
}