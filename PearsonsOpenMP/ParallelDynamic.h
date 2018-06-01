// Assignment 2 Comp 528, Robert Johnson, sgrjohn2@student.liverpool.ac.uk, 200962268
// Program to calculate Pearsons using omp dynamic scheduler
#include <omp.h>
void runParallelDynamic(int size, int thread_count, double timeTakenSerial)
{
    //Run Dynamic Version
    printf("------ParallelDynamic------\n");
    fflush(stdout);
    // start the timer for guided scheduler
    double t1 = omp_get_wtime();
    // allocate space for our array
    double *arrayA = malloc(size*sizeof(double));
    double *arrayB = malloc(size*sizeof(double));
    // fill our array in parallel using dynamic schedule
    #pragma omp parallel for schedule(dynamic) num_threads(thread_count)
    for (int i = 0; i < size; i++)
    {
        arrayA[i] = sin(i);
        arrayB[i] = sin(i+5);
    }
    // calculate the sum of each array
    double sumOfA = 0;
    double sumOfB = 0;
    // work out the sum in parallel using dynamic scheduling
    #pragma omp parallel for schedule(dynamic) reduction(+:sumOfB) reduction(+:sumOfA) num_threads(thread_count)
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
    // calculate part of SD and Pearsons in parallel
    #pragma omp parallel for schedule(static) reduction(+:standardDevA) reduction(+:standardDevB) reduction(+:pearsons) num_threads(thread_count)
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
    double t2 = omp_get_wtime() - t1;
    printf("sumA = %lf, standDevA = %lf, averageA = %lf\nsumB = %lf, standDevB = %lf, averageB = "
            "%lf\nPearsons = %lf\n", sumOfA, standardDevA, meanOfA, sumOfB, standardDevB, meanOfB, pearsons);
    fflush(stdout);
    printf("Time taken for dynamic Version was %lf\n",t2);
    if(timeTakenSerial!=0)
    {
        if(t2<timeTakenSerial)
        {
            printf("Dynamic was %.1lf%% quicker\n\n",((timeTakenSerial/t2)-1)*100);
        }
        else
        {
            printf("Dynamic was %.1lf%% slower\n\n",((t2/timeTakenSerial)-1)*100);
        }
    }
    else
    {
        printf("Cannot calculate speedup as serial was not ran or produced a time of 0.\n\n");
    }
}