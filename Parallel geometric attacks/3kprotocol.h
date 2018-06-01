// Robert Johnson 200962268
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <mpi.h> 
#include <string.h>
#include <limits.h>
int values[10];
extern int values[10];

// structure to create a neural network
struct NeuralNetwork 
{
  int** weights;
  int* hiddenLayerOutputs;
  int networkOutput;
} neuralNet;

// create a definition for booleans
typedef enum 
{
  true, false
} bool;

// pointers for methods.
bool runGeometricAttackKKKProtocolParallel(struct NeuralNetwork, struct NeuralNetwork, struct NeuralNetwork*, int**, int, int, int, int, int);
bool runKKKProtocolWithoutAttaacker(struct NeuralNetwork, struct NeuralNetwork, int**, int, int, int, int, int, int*, int*);
struct NeuralNetwork constructNeuralNetwork(int, int, int);
void initWeights(int**, int, int, int);
void updateWeights(int*, struct NeuralNetwork, int**, int, int, int);
int binaryRand(void);
int** getRandomInputs(int**, int, int);
int getMinInputSumNeuron(struct NeuralNetwork, int**, int, int);
int* getHiddenLayerOutputs(int*, struct NeuralNetwork, int**, int, int);
int getNetworkOutput(int*, struct NeuralNetwork, int**, int, int);
void freeMemoryForNetwork(struct NeuralNetwork, int, int);
char* printNetworkWeights(char*, struct NeuralNetwork, int, int, int, int);
bool compareNetworks(struct NeuralNetwork, struct NeuralNetwork ,int , int );
int numPlaces (int);

bool compareNetworks(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB ,int k , int n)
{
  bool isDifferent = false;
  for(int i = 0 ; i < k; i++)
  {
    for(int j = 0; j < n; j++)
    {
      if(neuralNetA.weights[i][j]!=neuralNetB.weights[i][j])
      {
        isDifferent = true;
      }
    }
  }
  return isDifferent;
}
/**
 * Simulates the geometric attack on the 3k protocol
 * @param neuralNetA - neuralNetA and neuralNetB are the normal communicating pair by which we wish to generate a common key.
 * @param neuralNetB
 * @param attackerNet - or neuralNetC which is for the attacker.
 * @param inputs - the kth 'row' of the 'two-dimensional' array contains the inputs to the kth neuron.
 * @param k - identifies the number of hidden neurons.
 * @param n - identifies the n of inputs into each hidden neurons. The total number of inputs to the network is therefore N = k*n.
 * @param l - is the bound (-l to l) on the range of values that can be assigned to the weights. It is proposed that the bigger the l, the more
 *      difficult it is to break the protocol.
 * @param syncThreshold - if the all the involved networks produce the same weights in 'syncThreshold' successive rounds,
 *             then we take it that the synchronisation is now stable and we can take the weights as final.
 * @param epochLimit  - in case the networks are taking too long to reach synchronisation stability, we set this limit on the number of rounds that 
 *          can be executed so that we don't run the simulation for ever. This limit will depend on the resources available to your simulation 
 *          environment.
 * @return true or false indicating whether synchronisation was reached or not. Synchronisation is reached when the attack succeeds i.e the attacker succeeds in synchronising its 
 *      network weights with that of network A and network B.
 */
bool runGeometricAttackKKKProtocolParallel(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork* attackerNet, int** inputs, int k, int n, int l, int syncThreshold, int epochLimit) 
{
  int s = 0;
  int epoch = 0;
  int* outputC = malloc(sizeof(int)*values[4]);
  // allocate space for hidden layer.
  int** hlOutputsGroupings = malloc(sizeof(int*)*values[4]);
  #pragma omp parallel for num_threads(values[4])
  for(int i = 0 ; i < values[4]; i++)
  {
    hlOutputsGroupings[i] = malloc(sizeof (int) * k);
  }
  while ((s < syncThreshold) && (epoch < epochLimit)) 
  {
    int outputA = getNetworkOutput(hlOutputsGroupings[0], neuralNetA, inputs, k, n);
    int outputB = getNetworkOutput(hlOutputsGroupings[0], neuralNetB, inputs, k, n);
    if(outputA==outputB)
    {
      //Update the weights of A and B using the anti-Hebbian learning rule.
      updateWeights(hlOutputsGroupings[0], neuralNetA, inputs, k, n, l);
      updateWeights(hlOutputsGroupings[0], neuralNetB, inputs, k, n, l);
      //Increase synchronisation count, s.
      s=s+1;
      #pragma omp parallel for num_threads(values[4])
      for(int i = 0; i < values[4]; i++)
      {
        outputC[i] = getNetworkOutput(hlOutputsGroupings[i], attackerNet[i], inputs, k, n);
        // if network C does not equal then
        if (outputA != outputC[i]) 
        {
          int kthHidden = getMinInputSumNeuron(attackerNet[i], inputs, k, n);
          //negate the output of the "minimum sum neuron" obtained above.
          attackerNet[i].hiddenLayerOutputs[kthHidden] = attackerNet[i].hiddenLayerOutputs[kthHidden] * (-1);
        }
        //For each C update the weight using the anti-hebbian learning rule
        updateWeights(hlOutputsGroupings[i], attackerNet[i], inputs, k, n, l);
      }
    }
    else
    {
      //Reset the synchronisation count - there was no synchronisation or sychronisation broke down in the round.
      s = 0;
    }
    //Get new random inputs for the next round.
    getRandomInputs(inputs, k, n);
    //Increment the round count. We will not run the protocol for ever - we will stop after a predefined number of rounds if 
    //synchronisation has not been reached by then.
    epoch ++;
  }
  // free output c
  free(outputC);
  // free all hidden output's
  #pragma omp parallel for num_threads(values[4])
  for(int i = 0 ; i < values[4]; i++)
  {
    free(hlOutputsGroupings[i]);
  }
  free(hlOutputsGroupings);
  //Did the above while loop stop because the synchronisation threshold was reached?
  if (s == syncThreshold) 
  {
    return true; // We have succesfully synchronised the network. The weights were the same for syncThreshold number of rounds!
  } 
  return false; //We've exceeded the epoch limit without succeeding in synchronising the network.
}

/**
 * Simulates the 3k protocol between two networks A and B. After the simulation the network weights for both network can be printed to show they are
 * synchronised. Use the utility function printNetworkWeights(...) in this library to print the network weights of network A and network B and attacker network.
 * 
 * @param neuralNetA - neuralNetA and neuralNetB are the normal communicating pair by which we wish to generate a common key.
 * @param neuralNetB
 * @param inputs - the kth 'row' of the 'two-dimensional' array contains the inputs to the kth neuron.
 * @param k - identifies the number of hidden neurons.
 * @param n - identifies the n of inputs into each hidden neurons. The total number of inputs to the network is therefore N = k*n.
 * @param l - is the bound (-l to l) on the range of values that can be assigned to the weights. It is proposed that the bigger the l, the more
 *      difficult it is to break the protocol.
 * @param syncThreshold - if the all the involved networks produce the same weights in 'syncThreshold' successive rounds,
 *             then we take it that the synchronisation is now stable and we can take the weights as final. 
 * 
 * @param epochLimit  - in case the networks are taking too long to reach synchronisation stability, we set this limit on the number of rounds that 
 *          can be executed so that we don't run the simulation for ever. This limit will depend on the resources available to your simulation 
 *          environment.
 * @param synchLimit  - Max synchronization required.
 * @param epochMax  - Max epochs required.
 * @return true or false indicating whether synchronisation was reached or not. 
 */
bool runKKKProtocolWithoutAttaacker(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, int** inputs, int k, int n, int l, int syncThreshold, int epochLimit, int* synchLimit, int* epochMax) 
{
  int s = 0;
  int epoch = 0;
  int* hlOutputs = malloc(sizeof (int) * k);
  while ((s < syncThreshold) && (epoch < epochLimit)) 
  {
    int outputA = getNetworkOutput(hlOutputs, neuralNetA, inputs, k, n);
    int outputB = getNetworkOutput(hlOutputs, neuralNetB, inputs, k, n);
    if(outputA==outputB)
    {
      updateWeights(hlOutputs, neuralNetA, inputs, k, n, l);
      updateWeights(hlOutputs, neuralNetB, inputs, k, n, l);
      s=s+1;
    }
    else
    {
      if(s>*synchLimit)
      {
        *synchLimit=s;
      }
      s = 0;
    }
    getRandomInputs(inputs, k, n);
    epoch ++;
  }
  free(hlOutputs);
  if(epoch>*epochMax)
  {
    *epochMax = epoch;
  }
  if (s == syncThreshold) 
  {
    return true;
  } 
  return false; 
}

/**
 * Constructs a new two layered neural network with k perceptrons, n inputs per perceptron and weight across each input generated randomly 
 * from the range -l to l.
 * @param k
 * @param n
 * @param l
 * @return the newly constructed neural network.
 */
struct NeuralNetwork constructNeuralNetwork(int k, int n, int l) 
{
    struct NeuralNetwork neuralNetwork;
    // Allocate memory block for the neural network weights and hiddent layer outputs.
    neuralNetwork.weights = malloc(sizeof (int*) * (k));
    //Allocate memory blocks for the hidden layer outputs.
    neuralNetwork.hiddenLayerOutputs = malloc(sizeof (int) * k);
    for (int i = 0; i < k; i++) 
    {
    neuralNetwork.weights[i] = malloc(sizeof (int) * n);
    for (int j = 0; j < n; j++) 
    {
      neuralNetwork.weights[i][j] = rand() % (2 * l + 1) - l;
    }
  }
  return neuralNetwork;
}

/**
 * Gets the neuron/perceptron whose sum of product of inputs and weights is the minimum, of all the perceptrons in the network.
 * @param neuralNetwork The network to be processed.
 * @param inputs to the network (not part of the NeuralNetwork structure).
 * @param k The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron.
 * @return  The index of the minimum input sum neuron.
 */
int getMinInputSumNeuron(struct NeuralNetwork neuralNetwork, int** inputs, int k, int n) 
{
  int sum = 0;
  int minSum = 0;
  int minSumNeuron = 0;
  // Calculate the sum of product of inputs and weights for each perceptron, and 
  // keep track of the minimum of all the perceptrons.
  for (int i = 0; i < k; i++)
  {
    for (int j = 0; j < n; j++) 
    {
      sum = sum + (inputs[i][j] * neuralNetwork.weights[i][j]);
    }
    //To get absolute value
    sum = abs(sum);
    // If current sum of product of inputs and weights is more than our previous
    // minimum, then we've got a new minimum.
    if ((minSum == 0) || (sum < minSum)) 
    { 
      minSum = sum;
      minSumNeuron = i;
    }
    sum = 0;  // Ready for next perceptron.
  }
  return minSumNeuron;
}

/**
 * Updates the weight vectors of a network using the anti-Hebbian learning rule: w(i) = w(i) - output * input(i) 
 * @param neuralNet The network whose weight is to be updated.
 * @param inputs The input vector containing the inputs to the network.
 * @param k  The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron in the network.
 * @param l  The upperbound (l) and lower bound (-l) of weight to be assigned.
 * 
 */
void updateWeights(int* hlOutputs, struct NeuralNetwork neuralNet, int** inputs, int k, int n, int l) 
{
  getHiddenLayerOutputs(hlOutputs, neuralNet, inputs, k, n);
  for (int i = 0; i < k; i++) 
  {
    for (int j = 0; j < n; j++) 
    {
      // Update the weight using anti-Hebbian learning rule.
      neuralNet.weights[i][j] = neuralNet.weights[i][j] + (hlOutputs[i]*inputs[i][j]);
      if (neuralNet.weights[i][j] < ((-1) * l)) 
      {
        neuralNet.weights[i][j] = (-1) * l;
      } 
      else if (neuralNet.weights[i][j] > l) 
      {
        neuralNet.weights[i][j] = l;
      }
    }
  }
}

// get the max number of chars a number can take up.
int numPlaces (int n) 
{
  if (n < 0) n = (n == INT_MIN) ? INT_MAX : -n;
  if (n < 10) return 1;
  if (n < 100) return 2;
  if (n < 1000) return 3;
  if (n < 10000) return 4;
  if (n < 100000) return 5;
  if (n < 1000000) return 6;
  if (n < 10000000) return 7;
  if (n < 100000000) return 8;
  if (n < 1000000000) return 9;
  return 10;
}

// method to create a string that we would like to print / send.
char* printNetworkWeights(char* prepend, struct NeuralNetwork neuralNet, int k, int n, int l, int printout) 
{
  // store the max space that a network could take up.
  char* str = malloc(sizeof(char)*(k*n*(values[5]+2)+(1*k)+50));
  // create an array to store the max value of a weight.
  char tmp[values[5]+2];
  // empty arrays.
  tmp[0] = '\0';
  str[0] = '\0';
  // add the prepend message to the string.
  strcat(str, prepend);
  // create a string which represents the neural network.
  for (int i = 0; i < k; i++) 
  {
    for (int j = 0; j < n; j++) 
    {
      sprintf(tmp, "%d, ", neuralNet.weights[i][j]);
      strcat(str, tmp);
    }
    strcat(str, "\n");
  }
  // if we want to print the string on the thread/node that we are running on then print.
  if(printout==0)
  {
    printf("%s",str);
    free(str);
  }
  return str;
}

/**
 * Generates a random number from the set {-1, 1}.
 * @return The generated random number.
 */
int binaryRand() 
{
  int randNum = rand();
  if (randNum % 2 == 0) 
  {
    return 1;
  } 
  else 
  { 
    return -1;
  }
}

/**
 * Generates random inputs value (each input value is either -1 or 1), to be used for a neural 
 * network with k perceptrons and n inputs per perceptron.
 * @param k
 * @param n
 * @return The input vector generated.
 */
int** getRandomInputs(int** inputs, int k, int n) 
{
  //generate and set the inputs.
  for (int i = 0; i < k; i++) 
  {
    for (int j = 0; j < n; j++) 
    {
      inputs[i][j] = binaryRand();
    }
  }
  return inputs;
}

/**
 * Trigger the hidden layer outputs for the supplied neural network, and then return the hidden
 * layer output vector.
 * @param neuralNet The network whose hidden layer outputs is to be triggered.
 * @param inputs  The inputs to the network.
 * @param k  The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron.
 * @return   The hidden layer outputs of the supplied network.
 */
int* getHiddenLayerOutputs(int* hlOutputs, struct NeuralNetwork neuralNet, int** inputs, int k, int n) 
{
  for (int i = 0; i < k; i++) 
  {
    int sum = 0;
    for (int j = 0; j < n; j++) 
    {
      sum = sum - (neuralNet.weights[i][j] * inputs[i][j]);
    }
    //Each hidden layer output must be either -1 or +1. We are interested in 
    // only the sign parity(negative or positive) of the output of each perceptron.
    if (sum <= 0) 
    {
      hlOutputs[i] = -1;
    } 
    else 
    {
      hlOutputs[i] = 1;
    }
  }
  return hlOutputs;
}

/**
 * Trigger the output of the neural network and return it.
 * @param neuralNet The network whose output is to be obtained.
 * @param inputs  The inputs to the network.
 * @param k  The number of perceptrons to the network.
 * @param n The number of inputs to each perceptron in the network.
 * @return The value of the output of the network.
 * 
 */
int getNetworkOutput(int* hlOutputs, struct NeuralNetwork neuralNet, int** inputs, int k, int n) 
{
  getHiddenLayerOutputs(hlOutputs, neuralNet, inputs, k, n);
  //Obtain the product of all the hidden layer outputs. Since each hidden layer
  //output is either 1 or -1, this product will give us a sign parity (positive or negative).
  int prod = 1;
  for (int i = 0; i < k; i++) 
  {
    prod = prod * (hlOutputs[i]);
  }
  return prod;
}

/**
 * Free up the memory allocated for a neural network.
 * @param neuralNet
 * @param k The number of perceptrons in the neural network.
 * @param n  The number of inputs to each perceptron in the network.
 */
void freeMemoryForNetwork(struct NeuralNetwork neuralNet, int k, int n) 
{
  // Free memory block for the weight vectors of the neural network;
  for (int i = 0; i < k; i++) 
  {
    free(neuralNet.weights[i]);
  }
  free(neuralNet.weights);
  //Free memory for the hidden layer outputs.
  free(neuralNet.hiddenLayerOutputs);
}