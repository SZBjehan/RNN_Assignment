import numpy as np

#################################################################################
#---------------- CTC Components ------------------------------------------------
# target, matrix, (target_len)                 target sequence
# logits, matrix, (input_len, len(Symbols))     predicted probabilities
# extSymbols, vector, (2*target_len+1)          output from extendign the target with blanks
# skipConnect, vector, (2*target_len+1)         boolean array containing skip connections
# alpha, matrix, (input_len, 2*target_len+1)    Forward probabilities
# beta, matrix, (input_len, 2*target_len+1)     Backward probabilities
# gamma, matrix, (input_len, 2*target_len+1)    Posterior probabilities
#################################################################################
class CTC(object):

    #DO NOT MODIFY this method
    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.
        Given an output sequence from an RNN/GRU, 
        we want to extend the target sequence with blanks, 
        where blank has been defined in the initialization.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
                    An array with same length as extSymbols to keep track of whether 
                    an extended symbol Sext(j) is allowed to connect directly to Sext(j-2) 
                    (instead of only to Sext(j-1)) or not. 
                    The elements in the array can be True/False or 1/0. 
                    This will be used in the forward and backward algorithms.

        ex: [0,0,0,1,0,0,0,1,0]
        """
        extended_symbols = []
        skip_connect = []

        # add necessary code
        # Update extSymbols #TODO
        # Update  skip_connect #TODO
        
        
        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros((T, S))

        # add necessary code
        # calculate alpha #TODO
    
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        # add necessary code
        # calculate beta #TODO

        
        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        # add necessary code
        # calculate gamma #TODO

        
        return gamma

#################################################################################
#---------------- CTC Loss Components ------------------------------------------------
# target, matrix, (batch_size, padded_target_len)       target sequence
# logits, matrix, (seqlength, batch_size, len(Symbols)) predicted probabilities
# input_lengths, vector, batch_size,    length of the inputs
# target_lengths, vector, batch_size,   length of the target
# loss, scalar,                         average divergence between posterior probability 
#                                       gamma and the input symbols y_t^r
# dY, matrix, (seqlength, batch_size, len(Symbols)  Derivative of divergence w.r.t.
#                                                   the input symbols at each time
#################################################################################
class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

                Initialize instance variables

        Argument(s)
                -----------
                BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

        
    # No need to modify
    def __call__(self, logits, target, input_lengths, target_lengths):
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

                Computes the CTC Loss by calculating forward, backward, and
                posterior proabilites, and then calculating the avg. loss between
                targets and predicted log probabilities

                The loss is average loss.  

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        # for b in range(B):
        #   add necessary code to update total_loss TODO
        #   need to store gammas and extSymbols, skip_connect, etc.
        

        return np.mean(total_loss)

    def backward(self):
        """

                CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative
                w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.zeros_like(self.logits)

        # for b in range(B):
        #   add necessary code to update total_loss TODO
        

        return dY
