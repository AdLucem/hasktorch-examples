import torch
import torch.nn as nn

# so first we define these parameters about our data
# and the number of iterations we want our network to go through
# (we do gradient descent with a fixed number of iterations)
INPUT_SIZE = 2
HIDDEN_SIZE = 2
OUTPUT_SIZE = 2
BATCH_SIZE = 1



def print_details(gru):

    print(gru.weight_ih)
    print(gru.weight_hh)
    print("--------------------------------")


if __name__ == "__main__":

    # -------------------- initialization bit ---------------

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

    # ----------------- forward bit ------------------------

    # first time step. hand-demonstrated iteration
    # for one time step only
    # at each iteration, the hidden state is updated
    hiddenState = gru(inputTensor, hiddenState)
    timeStates.append(hiddenState)
        
    
    outputTensor = hiddenState

    # ----------------- backprop bit -----------------------

    # print the initial weights of the cell
    print_details(gru)

    # arbitrary loss function
    loss = nn.L1Loss()

    # loss for our first timestep
    t1Loss = loss(outputTensor, targetTensor)
 
    # backprop the loss using pytorch's handy auto-backprop mechanism
    gru.zero_grad()
    t1Loss.backward()
    

    # and for the update, we'll use simple gradient descent
    # i.e: updatedWeights = weights - learning_rate * gradient
    # and a ridiculously large learning rate so that the change is noticeable 
    learningRate = 100

    gru.weight_ih = nn.Parameter(gru.weight_ih - (learningRate * gru.weight_ih.grad))
    gru.weight_hh = nn.Parameter(gru.weight_hh - (learningRate * gru.weight_hh.grad))

    # print the updated weights of the cell
    print_details(gru)    
