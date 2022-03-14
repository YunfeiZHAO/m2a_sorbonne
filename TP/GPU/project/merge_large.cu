#include <ctime>
#include <stdio.h>

#define NB 2
#define NTPB 1
// threads per block for the merge part
#define NTPB_Z 64

#define len_A 10
#define len_B 10

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

__global__ void partition(int *a, int *b, int *A_part, int *B_part, int lengthA, int lengthB){

    // *** Ne pas oublier le cas len_A == leb_B == 1 
    // *** Ne pas oublier NB >= min(len_A, len_B) 

	// A represents the biggest array
	int *A, *B, *ap, *bp, la, lb, offset, i;
	int Kx, Ky, Px, Py, Qx, Qy;

	if(lengthA > lengthB ){ 
        A = a;
		B = b;
		la = lengthA;
		lb = lengthB;
        ap = A_part;
        bp = B_part; 

	}else{
		B = a;
		A = b;
		lb = lengthA;
		la = lengthB;
        bp = A_part;
        ap = B_part;
	}

    int is_divisable = (len_A + len_B)%gridDim.x!=0;
    i =((len_A + len_B)/gridDim.x+ is_divisable ) * blockIdx.x;

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
				ap[blockIdx.x] = Qx;
                bp[blockIdx.x] = Qy;
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


void wrapper_partition(int *A, int *B, int *A_part, int *B_part) {
    int *aGPU, *bGPU, *a_partGPU, *b_partGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&aGPU, len_A*sizeof(int)));
    testCUDA(cudaMalloc(&bGPU, len_B*sizeof(int)));
    testCUDA(cudaMalloc(&a_partGPU, len_B*sizeof(NB)));
    testCUDA(cudaMalloc(&b_partGPU, len_B*sizeof(NB)));

    testCUDA(cudaMemcpy(aGPU, A, len_A*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(bGPU, B, len_B*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA(cudaMemcpy(a_partGPU, A_part, NB*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(b_partGPU, B_part, NB*sizeof(int), cudaMemcpyHostToDevice));
    
    //start of kernel
    partition<<<NB, NTPB>>>(aGPU, bGPU, a_partGPU, b_partGPU, len_A, len_B);

    testCUDA(cudaMemcpy(A_part, a_partGPU, (NB)*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(B_part, b_partGPU, (NB)*sizeof(int), cudaMemcpyDeviceToHost));

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    printf("Execution time: %f ms\n", TimerV);

    testCUDA(cudaFree(aGPU));
    testCUDA(cudaFree(bGPU));
    testCUDA(cudaFree(a_partGPU));
    testCUDA(cudaFree(b_partGPU));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
}

int main(void){
    int *A = (int*)malloc(len_A * sizeof(int));
    int *B = (int*)malloc(len_B * sizeof(int));
    int *A_part = (int*)malloc((NB) * sizeof(int));
    int *B_part = (int*)malloc((NB) * sizeof(int));

    for(int i=0; i<len_A; i++){
        A[i] = i;
	}
    for(int i=0; i<len_B; i++){
        B[i] = i+2;
    }
    wrapper_partition(A, B, A_part, B_part);
    for(int i = 0; i < NB; i++){
        printf("|%d: %d %d |",i, A_part[i], B_part[i]);
    }

    return 0;
}