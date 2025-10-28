import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            A list of symbols that can be predicted, except for the blank symbol.

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """
        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            The probability distribution over all symbols including the blank 
            symbol at each time step. The probability of blank for all time steps 
            is the first row of y_probs (index 0).
            Be careful with the batch size in all test cases. 
            If it is not 1, please make sure to incorporate batch_size. 

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        #add necessary code
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        

        return decoded_path, path_prob

