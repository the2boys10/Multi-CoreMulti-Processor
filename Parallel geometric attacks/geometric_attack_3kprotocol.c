// Robert Johnson 200962268
#include "3kprotocol.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <mpi.h> 
int EPOCH_LIMIT = 200000;
int SYNCHRONISATION_THRESHOLD = 2000;
int comm_sz; 
int my_rank;
int values[10];
int values2[2];
int textOff = 0;
void printOutNetworks(struct NeuralNetwork, struct NeuralNetwork, struct NeuralNetwork*);
// values[0] = k
// values[1] = n
// values[2] = l
// values[3] = how many tests
// values[4] = how many threads
// values[5] = char length of max int
// values[6] = time
// values[7] = sync max
// values[8] = epoch max
// values[9] = display neural networks 1 if we would like to 0 if we would like to hide.
// values[10] = #times out of 200 synchronized
int main(int argc, char *argv[]) 
{
  values[9] = 0; // by default hide neural networks
  values[3] = 1; // by default set the amount of tests to 1.
  values[8] = 0; // by default set the epoch max to 0.
  MPI_Init(NULL, NULL); // initialize Mpi
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  // if my rank is 0
  if(my_rank == 0)
  {
    // if we have an argument with the program then turn the text off else display the text.
    if(argc==2)
    {
      textOff = 1;
    }
    if(textOff==0) printf("Please enter your value for k\n");
    scanf("%d", &values[0]);
    if(textOff==0) printf("Please enter your value for n\n");
    scanf("%d", &values[1]);
    if(textOff==0) printf("Please enter your value for l\n");
    scanf("%d", &values[2]);
    if(textOff==0) printf("Please enter how many tests your would like to run\n");
    scanf("%d", &values[3]);
    if(textOff==0) printf("Please enter how many openmp threads you would like to run.\n");
    scanf("%d", &values[4]);
    if(textOff==0) printf("Running pre-tests to work out normal synchronization threshold as well as normal epoch's\n\n");
    // get the amount of chars that the max L will use.
    values[5] = numPlaces(values[2])+1;
  }
  // broadcast all values.
  MPI_Bcast(&values, 10, MPI_INT, 0, MPI_COMM_WORLD);
  // for 40 tests perform a serial version of the kkk algoriothm without an attacker
  if(my_rank==0)
  {
    for(int j = 0; j < 40; j++)
    {
      // set a random seed for srand depending on iteration and time.
      srand(time(NULL)+j);
      // create a random neural network for A and B.
      struct NeuralNetwork neuralNetA = constructNeuralNetwork(values[0], values[1], values[2]);
      struct NeuralNetwork neuralNetB = constructNeuralNetwork(values[0], values[1], values[2]);
      // allocate space for inputs.
      int** inputs = malloc(sizeof (int*) * values[0]);
      for (int i = 0; i < values[0]; i++) 
      {
        inputs[i] = malloc(sizeof (int) * values[1]);
      }
      // get random inputs.
      getRandomInputs(inputs, values[0], values[1]);
      // set out current sync maximum/ epoch maximum to 0
      int syncCheck = 0;
      int epochCheck = 0;
      // run the kkk algorithm without any attackers.
      bool status = runKKKProtocolWithoutAttaacker(neuralNetA, neuralNetB, inputs, values[0], values[1], values[2], SYNCHRONISATION_THRESHOLD, EPOCH_LIMIT, &syncCheck, &epochCheck);
      // free all inputs.
      for (int i = 0; i < values[0]; i++) 
      {
        free(inputs[i]);
      }
      free(inputs);
      // if the networks synchronized then set max epoch and max synch's
      if(status == true && compareNetworks(neuralNetA, neuralNetB, values[0], values[1]))
      {
        if(values[7]<syncCheck)
        {
          values[7]=syncCheck;
        }
        if(values[8]<epochCheck)
        {
          values[8]=epochCheck;
        }
      }
      // free neural networks.
      freeMemoryForNetwork(neuralNetA, values[0], values[1]);
      freeMemoryForNetwork(neuralNetB, values[0], values[1]);
    }
  }
  // if my rank is 0 and we would like to print instructions
  if(my_rank == 0 && textOff == 0)
  {
    // tell the user we have finished running pretests and tell them the max epoch's and synchronization needed, ask them that
    // this is ok or if they woud like to use their own values.
    printf("Finished running pre-tests, synchronization threshold = %d, epoch threshold = %d\n", values[7]+1,values[8]+20);
    printf("If you would like to use these values type \"1\" for yes else type \"0\" for no.\n");
    int reply;
    scanf("%d", &reply);
    if(reply==0)
    {
      printf("Please state your synchronization threshold\n");
      scanf("%d", &values[7]);
      printf("Please state your epoch threshold\n");
      scanf("%d", &values[8]);
    }
    printf("Would you like to view the neural network weights before the process and after, type \"1\" for yes and \"0\" for no.\n");
    scanf("%d", &values[9]);
    printf("\n");
  }
  // for the amount of tests we would like to carry out.
  for(int j = 0; j < values[3]; j++)
  {
    // if my_rank is 0 get the shared time value.
    if(my_rank==0)
    {
      values[6] = (int)time(NULL)+j;
    }
    // share the time value.
    MPI_Bcast(&values, 10, MPI_INT, 0, MPI_COMM_WORLD);
    // if we wouldn't like to run any tests break.
    if(values[3]==0)
    {
      break;
    }
    // set our seed to our shared time value plus the test number.
    srand(values[6]+j);
    // create an array to store all neural network c's
    struct NeuralNetwork* neuralNetC = malloc(sizeof(struct NeuralNetwork)*values[4]);
    // construct neural net A and B.
    struct NeuralNetwork neuralNetA = constructNeuralNetwork(values[0], values[1], values[2]);
    struct NeuralNetwork neuralNetB = constructNeuralNetwork(values[0], values[1], values[2]);
    // in parallel generate all neural networks C.
    #pragma omp parallel for num_threads(values[4])
    for(int i = 0; i < values[4]; i++)
    {
      srand(rand()+(my_rank+1));
      srand(rand()+(omp_get_thread_num()+1));
      neuralNetC[i] = constructNeuralNetwork(values[0], values[1], values[2]);
    }
    // if we would like to, print neural networks print neural network A B and C in order of ranks.
    if(values[9]==1)
    {
      if(my_rank==0) printf("---------- Test %d ----------\nBefore\n", j);
      printOutNetworks(neuralNetA,neuralNetB,neuralNetC);
    }
    // set our random seed back to the shared seed so all inputs genrated are the same.
    srand(values[6]+j+5);
    // malloc space for our input values.
    int** inputs = malloc(sizeof (int*) * values[0]);
    for (int i = 0; i < values[0]; i++) 
    {
      inputs[i] = malloc(sizeof (int) * values[1]);
    }
    // generate the random inputs.
    getRandomInputs(inputs, values[0], values[1]);
    // run the geometric attack on A, B and with maxSync+20 and maxEpoch+20.
    bool status = runGeometricAttackKKKProtocolParallel(neuralNetA, neuralNetB, neuralNetC, inputs, values[0], values[1], values[2], (values[7]+20), (values[8]+20));
    // if we would like to, print neural networks A, B and C in order of ranks.
    if(values[9]==1)
    {
      if(my_rank==0) printf("After\n");
      printOutNetworks(neuralNetA,neuralNetB,neuralNetC);
    }
    // set a value to sum all threads that managed to successfully attack A and B.
    int checkValue = 0;
    // for each neural network C check if C is the same as A,B and C, if they are then sum them for our omp group, afterwards free C.
    #pragma omp parallel for reduction(+:checkValue) num_threads(values[4])
    for(int i = 0; i < values[4]; i++)
    {
      if(status==true&&compareNetworks(neuralNetA,neuralNetB, values[0], values[1])&&compareNetworks(neuralNetA,neuralNetC[i], values[0], values[1]))
      {
        checkValue = 1;
      }
      freeMemoryForNetwork(neuralNetC[i], values[0], values[1]);
    }
    // create a global mpi variable to store the global score of attackers that managed to attack A, B and C
    int result = 0;
    // free the array of neural net c's
    free(neuralNetC);
    // reduce our mpi threads to a single result, if the result is greater than 1 then atleast 1 thread managed to crack A and B and therefore the attack was successful.
    MPI_Reduce(&checkValue, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // if my rank is 0 then check that A and B really were equal, if they were add one to values[2] which stores how many times A and B synched, if we successfully attacked as well add 1 to values[1].
    if(my_rank==0)
    {
      if(status==true&&compareNetworks(neuralNetA,neuralNetB, values[0], values[1]))
      {
        values2[0]++;
        if(result>0)
        {
          values2[1]++;
        }
      }
    }
    // free our inputs.
    for (int i = 0; i < values[0]; i++) 
    {
      free(inputs[i]);
    }
    free(inputs);
    // free network A and B.
    freeMemoryForNetwork(neuralNetA, values[0], values[1]);
    freeMemoryForNetwork(neuralNetB, values[0], values[1]);
  }
  // if my rank is 0 then print out the statistics.
  if(my_rank==0)
  {
      if(textOff==0) printf("---------- Statistics ----------\nValue of k = %d\nValue of n = %d\nValue of l = %d\n# of tests = %d\n# of attackers = %d\nChance of users synchronizing = %.2lf\nChance of attackers = %.2lf\nValue of Synchronization = %d\nValue of Epoch = %d\n\n",values[0],values[1],values[2],values[3],values[4]*comm_sz,(double)values2[0]*100/(double)values[3],(double)values2[1]*100/(double)values2[0],values[7],values[8]);
      else printf("%d,%d,%d,%d,%d,%.2lf,%.2lf,%d,%d\n",values[0],values[1],values[2],values[3],values[4]*comm_sz,(double)values2[0]*100/(double)values[3],(double)values2[1]*100/(double)values2[0],values[7],values[8]);
  }
  // finalize mpi.
  MPI_Finalize();
  return 0;
}

// method used to print out networks in the correct order.
void printOutNetworks(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork* neuralNetC)
{
  // if my rank is 0 then print out network A and B as well as its own network c's
  if(my_rank == 0)
  {
    printf("Network A\n");
    printNetworkWeights("",neuralNetA,values[0], values[1], values[2], 0);
    printf("\n");
    printf("Network B\n");
    printNetworkWeights("",neuralNetB,values[0], values[1], values[2], 0);
    printf("\n");
    // for each network C print out the network.
    for(int k = 0; k < values[4]; k++)
    {
      char prepend[20];
      prepend[0] = '\0';
      sprintf(prepend, "Network C%d\n", my_rank*values[4]+k+1);
      printNetworkWeights(prepend,neuralNetC[k],values[0], values[1], values[2],0);
      printf("\n");
    }
    // get ready for mpi threads to start sending their networks to be printed. print out in order of mpi rank.
    for(int k = 1 ; k < comm_sz; k++)
    {
      // for each mpi thread each thread will create values[4] attackers so wait for each network C.
      for(int p = 0; p < values[4]; p++)
      {
        char* printout[values[0]*values[1]*(values[5]+2)+(1*values[0])+50];
        MPI_Recv(&printout, (values[0]*values[1]*(values[5]+2)+(1*values[0])+50), MPI_CHAR, k , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s",printout);
        printf("\n");
      }
    }
  }
  else
  {
    // for each mpi thread that is not the host thread send the data of network C to the main thread in order of index value.
    for(int k = 0 ; k < values[4]; k++)
    {
      char prepend[20];
      sprintf(prepend, "Network C%d\n", (my_rank*values[4])+k+1);
      char* message = printNetworkWeights(prepend,neuralNetC[k],values[0], values[1], values[2], 1);
      MPI_Send(message, (values[0]*values[1]*(values[5]+2)+(1*values[0])+50), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
      free(message);
    }
  }
}
