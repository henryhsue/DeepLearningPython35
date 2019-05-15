# ipython% load startup.py

# note, install theano 1.04. It's not on any of the conda repo's, so use `pip install theano=1.04`
# https://github.com/pymc-devs/pymc3/issues/3340

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer 

training_data, validation_data, test_data = network3.load_data_shared()

mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)
