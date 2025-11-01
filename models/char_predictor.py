import numpy as np
import sys

# sys.path.append("mydnn")
from mydnn.gru_cell import *
from mydnn.linear import *


class CharacterPredictor(object):
    """CharacterPredictor class.

    This is the neural net that will run one timestep of the input
    You only need to implement the forward method of this class.
    This is to test that your GRU Cell implementation is correct when used as a GRU.

    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()

        """
        The network consists of a GRU Cell and a linear layer.
        We refer to the linear layer self.projection in the code 
        because it is just a linear transformation between the hidden state to the output state
        """
        # add necessary code
        # self.gru = TODO add a GRUCell
        # self.projection = TODO add a linear layer
        # self.num_classes = TODO # the number of classes being predicted from the Linear layer
        # self.hidden_dim = TODO
        # self.projection.W = TODO
        self.gru = GRUCell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        # self.projection.W = np.eye(self.num_classes, self.hidden_dim)   # (num_classes, hidden_dim)

        # self.projection.W = np.random.randn(num_classes, hidden_dim) * 0.01
        # self.projection.b = np.random.randn(num_classes, 1) * 0.01


    def init_rnn_weights(
        self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
    ):
        # DO NOT MODIFY
        self.gru.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
        )

    # DO NOT MODIFY
    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        hnext = self.gru(x, h)

        # add necessary code
        # logits = TODO
        hnext_2d = hnext.reshape(1, -1)
    
        # Pass through projection layer
        logits_2d = self.projection(hnext_2d)
        
        # Flatten back to 1D: (num_classes,)
        logits = logits_2d.flatten()
        
        return logits, hnext


def inference(net, inputs):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """

    # add necessary code
    # seq_len = TODO
    # logits = TODO
    # h = TODO
    seq_len = inputs.shape[0]
    logits = np.zeros((seq_len, net.num_classes))
    
    h = np.zeros(net.hidden_dim)
    
    for t in range(seq_len):
        # Get the input for this time step
        x_t = inputs[t]
        
        # Run one step of the network
        # l_t will be (num_classes,)
        # h will be (hidden_dim,) and gets passed to the next loop
        l_t, h = net(x_t, h)   
        
        # Store the logits
        logits[t] = l_t

    return logits
