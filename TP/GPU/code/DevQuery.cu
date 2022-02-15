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

	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/


	return 0;
}