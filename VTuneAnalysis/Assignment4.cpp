// #include "stdafx.h"
// Assignment 4 Comp 528, Robert Johnson, sgrjohn2@student.liverpool.ac.uk, 200962268
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
	// set thread count
	int thread_count = 8;
	// set array size
	int size = 5000000;
	// allocate space for our array
	double *arrayA = (double*)malloc(size * sizeof(double));
	double *arrayB = (double*)malloc(size * sizeof(double));
	// declare useful variables
	double sumOfA = 0;
	double sumOfB = 0;
	double standardDevA = 0;
	double standardDevB = 0;
	double pearsons = 0;
	double meanOfA;
	double meanOfB;
	// fill our array in parallel using dynamic schedule
#pragma omp parallel num_threads(thread_count)
	{
#pragma omp for schedule(dynamic,10000) 
		for (int i = 0; i < size; i++)
		{
			arrayA[i] = sin(i);
			arrayB[i] = sin(i + 5);
		}
		// calculate the sum of each array
		// work out the sum in parallel using dynamic scheduling
#pragma omp for schedule(dynamic,1000) reduction(+:sumOfB) reduction(+:sumOfA)
		for (int i = 0; i < size; i++)
		{
			sumOfB = sumOfB + arrayB[i];
			sumOfA = sumOfA + arrayA[i];
		}
		// work out the means of both arrays
#pragma omp single
		{
			meanOfA = sumOfA / size;
			meanOfB = sumOfB / size;
		}
		// initialize SD and pearsons to 0.
		// calculate part of SD and Pearsons in parallel
#pragma omp for schedule(dynamic,1000) reduction(+:standardDevA) reduction(+:standardDevB) reduction(+:pearsons)
		for (int i = 0; i < size; i++)
		{
			standardDevA += (arrayA[i] - meanOfA)*(arrayA[i] - meanOfA);
			standardDevB += (arrayB[i] - meanOfB)*(arrayB[i] - meanOfB);
			pearsons += ((arrayA[i] - meanOfA)*(arrayB[i] - meanOfB));
		}
	}
	// finish calulating SD and pearsons.
	standardDevA = sqrt(standardDevA / size);
	standardDevB = sqrt(standardDevB / size);
	pearsons = ((pearsons / size) / (standardDevA*standardDevB));
	// free it
	free(arrayA);
	free(arrayB);
	// print it out
	printf("sumA = %lf, standDevA = %lf, averageA = %lf\nsumB = %lf, standDevB = %lf, averageB = "
		"%lf\nPearsons = %lf\n", sumOfA, standardDevA, meanOfA, sumOfB, standardDevB, meanOfB, pearsons);
}
