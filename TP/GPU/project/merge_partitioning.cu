#include <ctime>
#include <stdio.h>

#define NB 500
#define NTPB 1024

#define len_A 1
#define len_B 10000

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
    ap = A_part;
    bp = B_part;
	if(lengthA > lengthB ){ 
        A = a;
		B = b;
		la = lengthA;
		lb = lengthB;
        ap = B_part;
        bp = A_part; 

	}else{
		B = a;
		A = b;
		lb = lengthA;
		la = lengthB;
        bp = B_part;
        ap = A_part;
    }

    int is_divisable = (len_A + len_B)%gridDim.x!=0;
    i = (len_A + len_B)/NB * (blockIdx.x);
    int part_len =  (len_A + len_B)/NB * (NB + 1);
    if(i > part_len-1)
        return;

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


__global__ void merge(int *a, int *b, int *A_part, int *B_part, int *M, int lengthA, int lengthB){
	// A represents the biggest array

	int *A, *B, la, lb, offset, *ap, *bp;
    int Kx, Ky, Px, Py, Qx, Qy;

    int start_x, start_y, end_x, end_y;
    int block = blockIdx.x;
    int i = threadIdx.x;

    A = a;
	B = b;
    ap = A_part;
    bp = B_part; 

    start_x = ap[block];
    start_y = bp[block];

    if(block ==0 ){
        start_x = 0;
        start_y = 0;
    }

    if(block < NB-1){
        end_x = ap[block+1];
        end_y = bp[block+1];
    }

    else{
        end_x = lengthA;
        end_y = lengthB;
    }

    int lengthA_c = abs(end_x - start_x);
    int lengthB_c = abs(end_y - start_y);

    __shared__ int A_copy[NTPB];
    __shared__ int B_copy[NTPB];

    if(i >= lengthA_c + lengthB_c){
		return; 
	}

	if(lengthA_c > lengthB_c){
        for(int l = 0; l< lengthA_c; l++)
            A_copy[l] = a[start_x + l];
        for(int l = 0; l< lengthB_c; l++)
            B_copy[l] = b[start_y + l];

        A = A_copy;
	    B = B_copy;
	    la = lengthA_c;
	    lb = lengthB_c;

	}else{
        for(int l = 0; l< lengthA_c; l++){
            A_copy[l] = a[start_x + l];
        }
        for(int l = 0; l< lengthB_c; l++)
            B_copy[l] = b[start_y + l];

    	B = A_copy;
		A = B_copy;
		lb = lengthA_c;
		la = lengthB_c;
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
					M[i + start_x + start_y]= A[Qy];
				}else{
					M[i + start_x + start_y]= B[Qx];
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

void wrapper_partition(int *A, int *B, int *A_part, int *B_part, int *C) {
    int *aGPU, *bGPU, *a_partGPU, *b_partGPU, *cGPU;
    float TimerV;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testCUDA(cudaMalloc(&aGPU, len_A*sizeof(int)));
    testCUDA(cudaMalloc(&bGPU, len_B*sizeof(int)));
    testCUDA(cudaMalloc(&a_partGPU, NB*sizeof(int)));
    testCUDA(cudaMalloc(&b_partGPU, NB*sizeof(int)));
    testCUDA(cudaMalloc(&cGPU, (len_A + len_B)*sizeof(int)));

    testCUDA(cudaMemcpy(aGPU, A, len_A*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(bGPU, B, len_B*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA(cudaMemcpy(a_partGPU, A_part, NB*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(b_partGPU, B_part, NB*sizeof(int), cudaMemcpyHostToDevice));

    //start of kernel
    partition<<<NB, 1>>>(aGPU, bGPU, a_partGPU, b_partGPU, len_A, len_B);
    merge<<<NB, NTPB>>>(aGPU, bGPU, a_partGPU, b_partGPU, cGPU, len_A, len_B);

    testCUDA(cudaMemcpy(C, cGPU, (len_A + len_B)*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(A_part, a_partGPU, (NB)*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(B_part, b_partGPU, (NB)*sizeof(int), cudaMemcpyDeviceToHost));

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    printf("Execution time: %f ms\n", TimerV);

    testCUDA(cudaFree(aGPU));
    testCUDA(cudaFree(bGPU));
    testCUDA(cudaFree(cGPU));
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

    int *C = (int*)malloc((len_A + len_B) * sizeof(int));

    for(int i=0; i<len_A; i++){
        A[i] = i;
	}
    for(int i=0; i<len_B; i++){
        B[i] = i;
    }
    wrapper_partition(A, B, A_part, B_part, C);
    for(int i = 0; i < NB; i++){
        printf("|%d: %d %d | blah\n",i, A_part[i], B_part[i]);
    }
    printf("\n");
    for(int i = 0; i < len_A + len_B; i++){
        printf("| %d |", C[i], i);
    }

    return 0;
}