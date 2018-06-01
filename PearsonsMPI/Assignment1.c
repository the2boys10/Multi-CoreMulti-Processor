// Assignment 1 Comp 528, Robert Johnson, sgrjohn2@student.liverpool.ac.uk, 200962268
// Program to test and compare the runtime of calculating pearsons coefficient
// in parallel and serial, as well as comparing the differences in time between
// padded and unpadded versions of parallel implamentation.

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <malloc.h>

// create inline functions to each method.
inline void runSerial();
inline void runParallelWithPadding();
inline void runParallelWithScatterV();


// initialize rank, numberofprocessors, ints for sendCount and recievecounts
int numProc, rank, sendCount, recvcount;
// set the size of all arrays
const int SIZE = 2000000;
// store the times of each process
double timeTakenSerial, t1, t2;
// create double arrays.
double *arrayA;
double *arrayB;



int main()
{
	// initialize mpi
	MPI_Init(NULL,NULL);
	// assign rank and processor number
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);
								//Run Serial Version
	if(rank==0)
	{
		printf("\nSerial\n");
		fflush(stdout);
		runSerial();
	}
								//Run Padded Version
	if(rank==0)
	{
		printf("\nPadded\n");
		fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	runParallelWithPadding();
								//Run ScatterV Version
	if(rank==0)
	{
		printf("\nScatterV\n");
		fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	runParallelWithScatterV();
	if(rank==0)
	{
		printf("\n");
	}
	MPI_Finalize();
	return 0;
}

//Serial version.
void runSerial()
{
	// allocate space for our array
	double *arrayA = malloc(SIZE*sizeof(double));
	double *arrayB = malloc(SIZE*sizeof(double));
	// fill our array
	for (int i = 0; i < SIZE; i++)
	{
		arrayA[i] = sin(i);
		arrayB[i] = sin(i+5);
	}
	// start timing
	t1 = MPI_Wtime();
	// calculate the sum of each array
	double sumOfA = 0;
	double sumOfB = 0;
    for (int i = 0; i < SIZE; i++)
    {
    	double sumOfBTemp = sumOfB;
    	double sumOfATemp = sumOfA;
        sumOfB = sumOfB + arrayB[i];
    	sumOfA = sumOfA + arrayA[i];
	}
	// work out the means of both arrays
	double meanOfA = sumOfA/SIZE;
	double meanOfB = sumOfB/SIZE;
	// initialize SD and pearsons to 0.
	double standardDevA = 0;
	double standardDevB = 0;
	double pearsons = 0;
	// calculate part of SD and Pearsons
	for(int i = 0; i < SIZE; i++)
	{
		standardDevA = standardDevA + pow((arrayA[i] - meanOfA),2);
		standardDevB = standardDevB + pow((arrayB[i] - meanOfB),2);
		pearsons = pearsons + ((arrayA[i]-meanOfA)*(arrayB[i]-meanOfB));
	}
	// finish calulating SD and pearsons.
	standardDevA = sqrt(standardDevA/SIZE);
	standardDevB = sqrt(standardDevB/SIZE);
	pearsons = ((pearsons/SIZE)/(standardDevA*standardDevB));
	// free it
	free(arrayA);
	free(arrayB);
	// print it out
	printf("sumA = %lf, standDevA = %lf, averageA = %lf\nsumB = %lf, standDevB = %lf, averageB = "
		"%lf\nPearsons = %lf\n", sumOfA, standardDevA, meanOfA, sumOfB, standardDevB, meanOfB, pearsons);
	t2 = MPI_Wtime();
	printf("Time taken was %1.20f\n",t2-t1);
	fflush(stdout);
	timeTakenSerial = t2-t1;
}

// Method to calculate the pearsons coefficient using Padding
void runParallelWithPadding()
{
	// How much are we going to buffer the array
	int bufferCount = numProc - SIZE%numProc;
	if (bufferCount == numProc)
	{
		bufferCount = 0;
	}
	if(rank==0)
	{
		// assign the un modified array to size determined and fill it(decided to modify
		// it later to allow for a better comparison of each method)
		arrayA = malloc(SIZE*sizeof(double));
		arrayB = malloc(SIZE*sizeof(double));
		for(int j = 0; j < SIZE; j++)
		{
			arrayA[j] = sin(j);
			arrayB[j] = sin(j+5);
		}
		// start timer
		t1 = MPI_Wtime();
		// modify the arrays so that they are padded and assign NaN to extra space.
		if(bufferCount>0)
		{
			arrayA = (double *)realloc(arrayA, (SIZE+bufferCount)*sizeof(double));
			arrayB = (double *)realloc(arrayB, (SIZE+bufferCount)*sizeof(double));
		}
		for(int i = SIZE; i < SIZE+bufferCount; i++)
		{
			arrayA[i] = NAN;
			arrayB[i] = NAN;
		}
	}
	// set the amount we send to each process to be the padded size / number of processors
	sendCount = (SIZE+bufferCount)/numProc;
	recvcount = (SIZE+bufferCount)/numProc;
	// create space for the recieved arrays
	double *recvbufA = malloc(recvcount*sizeof(double));
	double *recvbufB = malloc(recvcount*sizeof(double));
	// scatter the data
	MPI_Scatter(arrayA,sendCount,MPI_DOUBLE,recvbufA,recvcount,
         MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Scatter(arrayB,sendCount,MPI_DOUBLE,recvbufB,recvcount,
         MPI_DOUBLE,0,MPI_COMM_WORLD);
	// create local array to store the local sums of both arrays
	double localSums[2];
	localSums[0]=0;
	localSums[1]=0;
	// as long as we don't find NaN keep adding, else assign the amount of actual number it was sent to be i.
	for (int i = 0; i < sendCount; i++)
	{
		if(!isnan(recvbufA[i]))
		{
			localSums[0] += recvbufA[i];
			localSums[1] += recvbufB[i];
		}
		else
		{
			sendCount = i;
		}
	}
	// create array to store the global sum and reduce so we have the sum
	double globalSums[2];
	MPI_Allreduce(localSums,globalSums,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	// work out averages on each process
	double averages[2];
	averages[0] = globalSums[0]/SIZE;
	averages[1] = globalSums[1]/SIZE;
	// create a local variable to store local standard deviation as well as local pearsons.
	double finalResults[3];
	for (int i = 0; i < sendCount; i++)
	{
		double TempA = recvbufA[i]-averages[0];
		double TempB = recvbufB[i]-averages[1];
		finalResults[0] += pow(TempA,2);
		finalResults[1] += pow(TempB,2);
		finalResults[2] += TempA*TempB;
	}
	// reduce to global standard deviation and pearsons
	double globalFinalResults[3];
	MPI_Reduce(finalResults,globalFinalResults,3,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	// free our process buffers
	free(recvbufA);
	free(recvbufB);
	if(rank==0)
	{
		// calculate global standard deviation and pearsons.
		double standDevA = sqrt(globalFinalResults[0]/SIZE);
		double standDevB = sqrt(globalFinalResults[1]/SIZE);
		double pearsons = (globalFinalResults[2]/SIZE)/(standDevA*standDevB);
		// free global arrays.
		free(arrayA);
		free(arrayB);
		t2 = MPI_Wtime();
		// print out results
		printf("sumA = %lf, standDevA = %lf, averageA = %lf\nsumB = %lf, standDevB = %lf, averageB = "
			"%lf\nPearsons = %lf\n", globalSums[0], standDevA, averages[0], globalSums[1], standDevB, averages[1], pearsons);
		printf("Overall time taken is %1.20f\n",t2-t1);
		fflush(stdout);
		if(t2-t1 > timeTakenSerial)
		{
			printf("Serial was %d%% quicker\n",(int)floor(((t2-t1)/timeTakenSerial)*100)-100);
			fflush(stdout);
		}
		else
		{
			printf("Parallel was %d%% quicker\n",(int)floor((timeTakenSerial/(t2-t1))*100)-100);
			fflush(stdout);
		}
	}
}



// Method to calculate the pearsons coefficient using ScatterV
void runParallelWithScatterV()
{
	// create arrays for displacement value's and send amounts.
	int displs[numProc];
	int send_counts[numProc];
	// Only create space for arrays on process 0 as well as filling them
	if(rank==0)
	{
		arrayA = malloc(SIZE*sizeof(double));
		arrayB = malloc(SIZE*sizeof(double));
		for(int j = 0; j < SIZE; j++)
		{
			arrayA[j] = sin(j);
			arrayB[j] = sin(j+5);
		}
		t1 = MPI_Wtime();
	}
	// create a temporary variable to store the amount of process's
	int numProcTemp = numProc;
	// store the size of the array in temp
	int sizeTemp = SIZE;
	// set the displacement for process 0 to 0.
	displs[0] = 0;
	// set the amount that process 0 recieves to the floor of sizeTemp/numProcTemp
	send_counts[0] = sizeTemp/numProcTemp;
	// for each processor
	for(int i = 1; i < numProc; i++)
	{
		// find out the displacement of process i as well as how much it sends
		displs[i] = displs[i-1] + send_counts[i-1];
		sizeTemp -= send_counts[i-1];
		// we have 1 less processor to give tasks to
		numProcTemp--;
		// set the amount of data to send the process
		send_counts[i] = sizeTemp/numProcTemp;
	}
	// create a recieve buffer for all process's
	double *recvbufA = malloc(send_counts[rank]*sizeof(double));
	double *recvbufB = malloc(send_counts[rank]*sizeof(double));
	// scatter the initial arrays.
    MPI_Scatterv(arrayA, send_counts, displs, MPI_DOUBLE, recvbufA , send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(arrayB, send_counts, displs, MPI_DOUBLE, recvbufB , send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // create local array to store the local sums of both arrays
	double localSums[2];
	localSums[0] = 0;
	localSums[1] = 0;
	for (int i = 0; i < send_counts[rank]; i++)
	{
		localSums[0] += recvbufA[i];
		localSums[1] += recvbufB[i];
	}
	// create a global array to store the global sums, give it to each process.
	double globalSums[2];
	MPI_Allreduce(localSums,globalSums,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	// find out the average on each process.
	double averages[2];
	averages[0] = globalSums[0]/SIZE;
	averages[1] = globalSums[1]/SIZE;
	// create a local variable to store local standard deviation as well as local pearsons.
	double finalResults[3];
	finalResults[0]=0;
	finalResults[1]=0;
	finalResults[2]=0;
	for (int i = 0; i < send_counts[rank]; i++)
	{
		double TempA = recvbufA[i]-averages[0];
		double TempB = recvbufB[i]-averages[1];
		finalResults[0] += pow(TempA,2);
		finalResults[1] += pow(TempB,2);
		finalResults[2] += TempA*TempB;
	}
	// reduce to global standard deviation and pearsons
	double globalFinalResults[3];
	MPI_Reduce(finalResults,globalFinalResults,3,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	// free our process buffers
	free(recvbufA);
	free(recvbufB);
	if(rank==0)
	{
		// calculate global standard deviation and pearsons.
		double standDevA = sqrt(globalFinalResults[0]/SIZE);
		double standDevB = sqrt(globalFinalResults[1]/SIZE);
		double pearsons = (globalFinalResults[2]/SIZE)/(standDevA*standDevB);
		// free global arrays.
		free(arrayA);
		free(arrayB);
		t2 = MPI_Wtime();
		// print out results
		printf("sumA = %lf, standDevA = %lf, averageA = %lf\nsumB = %lf, standDevB = %lf, averageB = "
			"%lf\nPearsons = %lf\n", globalSums[0], standDevA, averages[0], globalSums[1], standDevB, averages[1], pearsons);
		printf("Overall time taken is %1.20f\n",t2-t1);
		fflush(stdout);
		if(t2-t1 > timeTakenSerial)
		{
			printf("Serial was %d%% quicker\n",(int)floor(((t2-t1)/timeTakenSerial)*100)-100);
			fflush(stdout);
		}
		else
		{
			printf("Parallel was %d%% quicker\n",(int)floor((timeTakenSerial/(t2-t1))*100)-100);
			fflush(stdout);
		}
	}
}