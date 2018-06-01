// Assignment 2 Comp 528, Robert Johnson, sgrjohn2@student.liverpool.ac.uk, 200962268
// Program to calculate Pearsons using omp serial scheduler
#include <time.h>
double runSerial(int size)
{
    //Run Serial Version
    printf("------Serial------\n");
    fflush(stdout);
    // start the timer for static scheduler
    clock_t t1 = clock();
    // allocate space for our array
    double *arrayA = malloc(size*sizeof(double));
    double *arrayB = malloc(size*sizeof(double));
    // fill our array
    for (int i = 0; i < size; i++)
    {
        arrayA[i] = sin(i);
        arrayB[i] = sin(i+5);
    }
    // calculate the sum of each array
    double sumOfA = 0;
    double sumOfB = 0;
    for (int i = 0; i < size; i++)
    {
            sumOfB = sumOfB + arrayB[i];
            sumOfA = sumOfA + arrayA[i];
    }
    // work out the means of both arrays
    double meanOfA = sumOfA/size;
    double meanOfB = sumOfB/size;
    // initialize SD and pearsons to 0.
    double standardDevA = 0;
    double standardDevB = 0;
    double pearsons = 0;
    // calculate part of SD and Pearsons
    for(int i = 0; i < size; i++)
    {
        standardDevA += (arrayA[i] - meanOfA)*(arrayA[i] - meanOfA);
        standardDevB += (arrayB[i] - meanOfB)*(arrayA[i] - meanOfA);
        pearsons += ((arrayA[i]-meanOfA)*(arrayB[i]-meanOfB));
    }
    // finish calulating SD and pearsons.
    standardDevA = sqrt(standardDevA/size);
    standardDevB = sqrt(standardDevB/size);
    pearsons = ((pearsons/size)/(standardDevA*standardDevB));
    // free it
    free(arrayA);
    free(arrayB);
    // print it out
    double t2 = (double)(clock() - t1) / CLOCKS_PER_SEC;
    printf("sumA = %lf, standDevA = %lf, averageA = %lf\nsumB = %lf, standDevB = %lf, averageB = "
            "%lf\nPearsons = %lf\n", sumOfA, standardDevA, meanOfA, sumOfB, standardDevB, meanOfB, pearsons);
    fflush(stdout);
    printf("Time taken for serial Version was %lf\n\n\n",t2);
    return t2;
}