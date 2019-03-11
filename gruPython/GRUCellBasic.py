import torch
import torch.nn as nn

# so first we define these parameters about our data
# and the number of iterations we want our network to go through
# (we do gradient descent with a fixed number of iterations)
INPUT_SIZE = 2
HIDDEN_SIZE = 2
OUTPUT_SIZE = 2
BATCH_SIZE = 1
TIME_STEPS = 10


def print_details(gru):

    print(gru.weight_ih)
    print(gru.weight_hh)
    print("--------------------------------")


def singleLayerGRU(input, initHidden, timesteps):

    # defines our GRUCell object first
    # this is a single GRU cell for an input vector
    # so, a single RNN layer
    gru = nn.GRUCell(INPUT_SIZE, HIDDEN_SIZE)

    # make a container to capture the hidden states
    # of our cell over time
    hiddenStateAtTime = []

    # append initial(random) hidden state to the container
    hiddenStateAtTime.append(hiddenState)

    # iterate over `n` time steps
    for i in range(timesteps):

        # at each iteration, the hidden state is updated
        hiddenState = gru(inputTensor, hiddenState)

        # the current hidden state is appended to the
        # list of hidden states indexed by timestep
        hiddenStateAtTime.append(hiddenState)


if __name__ == "__main__":

    # defines a GRUCell object
    gru = nn.GRUCell(INPUT_SIZE, HIDDEN_SIZE)

    # output: stores the h_t of the GRU for all time steps
    output = []

    inputTensor = torch.randn(BATCH_SIZE, INPUT_SIZE)
    hiddenState = torch.randn(BATCH_SIZE, HIDDEN_SIZE, requires_grad=True)
    targetTensor = torch.randn(BATCH_SIZE, HIDDEN_SIZE)

    # since GRUs don't have an output per se, we take the 'outputTensor'
    # to be the hidden state at the last timestep
    lastHiddenState = torch.randn(BATCH_SIZE, OUTPUT_SIZE)

    timeStates = []

    #print("--------------------------------")
    #print(gru.weight_ih)
    #print(gru.weight_hh)
    #print("--------------------------------")

    # first time step
    # at each iteration, the hidden state is updated
    hiddenState = gru(inputTensor, hiddenState)
    timeStates.append(hiddenState)
    print(gru.weight_ih)
    print(gru.weight_hh)
    print("--------------------------------")
        
    
    outputTensor = hiddenState

    # arbitrary loss function
    loss = nn.L1Loss()

    # loss for our first timestep
    t1Loss = loss(outputTensor, targetTensor)
    # backprop the loss using pytorch's handy auto-backprop mechanism

    gru.zero_grad()
    t1Loss.backward()
    

    # print the updated gradients
    #print(gru.weight_ih.grad)
    #print(gru.weight_hh.grad)
    #print("--------------------------------")

    # and for the update, we'll use simple gradient descent
    # i.e: updatedWeights = weights - learning_rate * gradient
    # and a ridiculously large learning rate so that the change is noticeable 
    learningRate = 100
    gru.weight_ih = nn.Parameter(gru.weight_ih - (learningRate * gru.weight_ih.grad))
    gru.weight_hh = nn.Parameter(gru.weight_hh - (learningRate * gru.weight_hh.grad))

    # print the updated weights
    print(gru.weight_ih)
    print(gru.weight_hh)
    print("--------------------------------")
    
