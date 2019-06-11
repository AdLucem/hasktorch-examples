import torch.nn as nn
import torch

def print_details(rnncell):

    print(rnncell.weight_ih)
    print(rnncell.weight_hh)
    print("***")

rnn = nn.RNNCell(2, 2)
inp = torch.randn(1, 2)
init_hidden = torch.randn(1, 2)
out = torch.randn(1, 2)

ff_1 = rnn(inp, init_hidden)

print(ff_1)

# print the initial weights of the cell
print_details(rnn)

# arbitrary loss function
loss = nn.L1Loss()

# loss for our first timestep
t1Loss = loss(out, inp)

# backprop the loss using pytorch's handy auto-backprop mechanism
rnn.zero_grad()
out.backward()
"""
# and for the update, we'll use simple gradient descent
# i.e: updatedWeights = weights - learning_rate * gradient
# and a ridiculously large learning rate so that the change is noticeable 
learningRate = 100

rnn.weight_ih = nn.Parameter(rnn.weight_ih - (learningRate * rnn.weight_ih.grad))
rnn.weight_hh = nn.Parameter(rnn.weight_hh - (learningRate * rnn.weight_hh.grad))

# print the updated weights of the cell
print_details(rnn) 
"""