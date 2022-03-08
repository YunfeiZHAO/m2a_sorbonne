#include <stdio.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void empty_k(void){

	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/

}

int main (void){

	empty_k<<<1,1>>>();

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("The number of devices available %d\n", deviceCount);

	int device;
	
	device = 0;
	//for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		testCUDA(cudaGetDeviceProperties(&deviceProp, device));

		printf("Name of my GPU %s\n", deviceProp.name);
		printf("size of global memory in bytes %zd\n", deviceProp.totalGlobalMem);
		printf("size of shared memory in bytes %zd\n", deviceProp.sharedMemPerBlock);
		printf("number of registers %d\n", deviceProp.regsPerBlock);
		printf("maximum number of threads per block %d\n", deviceProp.maxThreadsPerBlock);

		printf("maximum number of threads %d x %d x %d\n", deviceProp.maxThreadsDim[0], 
				deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

		printf("maximum number of threads %d x %d x %d\n", deviceProp.maxGridSize[0], 
				deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);


		printf("Device %d has compute capability %d.%d.\n",
		device, deviceProp.major, deviceProp.minor);
	//}


	return 0;
}