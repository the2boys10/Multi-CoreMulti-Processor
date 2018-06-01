// Assignment 2 Comp 528, Robert Johnson, sgrjohn2@student.liverpool.ac.uk, 200962268
// Program to test and compare the runtime of calculating pearsons coefficient
// in parallel and serial, as well as comparing the differences in time between
// different schedualing techiniques
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include "Serial.h"
#include "ParallelDynamic.h"
#include "ParallelGuided.h"
#include "ParallelStatic.h"
// Main method to run all different kinds of parallel implementations and compare them to the serial version's time
// Feel free to comment out whichever parallel methods you do not wish to run along with their corrisponding header.
// Guided seems to give the best results.
// Program must have runSerial uncommented to get accurate runtime for comparison.
int main(int argc, char *argv[])
{
    // set thread count
    int thread_count = 1; 
    // set array size
    int size = 5000000;
    // ask user for thread count.
    printf("Enter the amount of threads:");
    scanf("%d", &thread_count);
    printf("\n\n");
    double timeTakenSerial = 0;
    // get the time taken for serial process to complete
    timeTakenSerial = runSerial(size);
    // run guided and compare
    runParallelGuided(size, thread_count, timeTakenSerial);
    // run static and compare
    runParallelStatic(size, thread_count, timeTakenSerial);
    // run dynamic and compare
    runParallelDynamic(size, thread_count, timeTakenSerial);
}
