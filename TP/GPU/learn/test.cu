#include <stdio.h>

int main(void) {
    int a = 1;
    int *p = &a;
    printf("a: %d\n", a);
    printf("adresse a: %d\n", &a);
    printf("pointer a: %d\n", p);
    printf("adresse pointer: %d\n", &p);
}