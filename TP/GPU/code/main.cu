#include <stdio.h>
#include "timer.h"


void addVect(int *a, int *b, int *c, int length){

	int i;

	for(i=0; i<length; i++){
		c[i] = a[i] + b[i];
	}
}


int main (void){

	// Variables definition
	int *a, *b, *c;
	int i;
	
	// Length for the size of arrays
	int length = 1e8;

	Timer Tim;							// CPU timer instructions

	// Memory allocation of arrays 
	a = (int*)malloc(length*sizeof(int));
	b = (int*)malloc(length*sizeof(int));
	c = (int*)malloc(length*sizeof(int));

	// Setting values
	for(i=0; i<length; i++){
		a[i] = i;
		b[i] = 9*i;
	}

	Tim.start();						// CPU timer instructions

	// Executing the addition 
	addVect(a, b, c, length);

	Tim.add();							// CPU timer instructions

	// Displaying the results to check the correctness 
	for(i=length-50; i<length-45; i++){
		printf(" ( %i ): %i\n", a[i]+b[i], c[i]);
	}

	printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
		   (float)Tim.getsum());			// CPU timer instructions

	// Freeing the memory
	free(a);
	free(b);
	free(c);

	return 0;
}