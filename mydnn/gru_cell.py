import numpy as np
from mydnn.activation import *

# 
# This class is going to replicate a portion of the torch.nn.GRUCell interface
# 
#################################################################################
# input_size: H_in, scalar, the number of expected features in the input x
# hidden_size: H_out, scalar, the number of features in the hidden state
# -------------------------------------------------------------------------------
# x: x_t,               vector, H_in, observation at the current time step
# h_prev_t: h_{t-1},    vector, H_out, hidden state at the previous time step
# -------------------------------------------------------------------------------
# W_rx: maxtrix, H_out x H_in,  weight matrix for input (for reset gate)
# W_zx: maxtrix, H_out x H_in,  weight matrix for input (for update gate)
# W_nx: maxtrix, H_out x H_in,  weight matrix for input (for candidate hidden state)
# W_rh: maxtrix, H_out x H_out,  weight matrix for hidden state (for reset gate)
# W_zh: maxtrix, H_out x H_out,  weight matrix for hidden state (for update gate)
# W_nh: maxtrix, H_out x H_out,  weight matrix for hidden state (for candidate hidden state)
# -------------------------------------------------------------------------------
# b_rx: vector,  H_out,         bias for input (for reset gate)
# b_zx: vector,  H_out,         bias for input (for update gate)
# b_nx: vector,  H_out,         bias for input (for candidate hidden state)
# b_rh: vector,  H_out,         bias for hidden state (for reset gate)
# b_zh: vector,  H_out,         bias for hidden state (for update gate)
# b_nh: vector,  H_out,         bias for hidden state (for candidate hidden state)
# -------------------------------------------------------------------------------
# ----USED in BACKWARD calculation ----------------------------------------------
# delta: vector,   H_out,      gradient of loss w.r.t. h_t
# dx: vector,       H_in,      gradient of loss w.r.t. x_t
# dh_prev_t: vector, H_out,    gradient of loss w.r.t. h_{t-1}
#
# dW_rx: maxtrix, H_out x H_in,  gradient of loss w.r.t. W_rx
# dW_zx: maxtrix, H_out x H_in,  gradient of loss w.r.t. W_zx
# dW_nx: maxtrix, H_out x H_in,  gradient of loss w.r.t. W_nx
# dW_rh: maxtrix, H_out x H_out, gradient of loss w.r.t. W_rh
# dW_zh: maxtrix, H_out x H_out, gradient of loss w.r.t. W_zh
# dW_nh: maxtrix, H_out x H_out, gradient of loss w.r.t. W_nh
#
# db_rx: vector,  H_out,         gradient of loss w.r.t. b_rx
# db_zx: vector,  H_out,         gradient of loss w.r.t. b_zx
# db_nx: vector,  H_out,         gradient of loss w.r.t. b_nx
# db_rh: vector,  H_out,         gradient of loss w.r.t. b_rh
# db_zh: vector,  H_out,         gradient of loss w.r.t. b_zh
# db_nh: vector,  H_out,         gradient of loss w.r.t. b_nh
#################################################################################
class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    # DO NOT change this method
    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.
        In forward, we calculate h_t. 

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Attributes
        -----
            Forward stores variables x, hidden, r, z, and n. 

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # add necessary code
        #self.r = TODO
        #self.z = TODO
        #self.n = TODO
        #h_t = TODO
        
        # Calculate reset gate
        r_linear = self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh
        self.r = self.r_act.forward(r_linear) # <-- Fix 1

        # Calculate update gate
        z_linear = self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh
        self.z = self.z_act.forward(z_linear) # <-- Fix 1

        # Calculate candidate hidden state
        n_linear = self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh)
        self.n = self.h_act.forward(n_linear) # <-- Fix 1

        # Calculate new hidden state
        h_t = (1 - self.z) * self.n + self.z * self.hidden # <-- Fix 2: Using self.hidden

        # Store for backward pass
        self.h_t = h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.
    
        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        dx = np.zeros((self.d,))
        dh_prev_t = np.zeros((self.h,))


        #add necessary code
        #dh_prev_t = TODO
        #dn = TODO
        #dz = TODO
        #dr = TODO
        #self.dWrh = TODO
        #self.dWzh = TODO
        #self.dWnh = TODO

        #self.dWrx = TODO
        #self.dWzx = TODO
        #self.dWnx = TODO
        
        #self.dbrx = TODO
        #self.dbzx = TODO
        #self.dbnx = TODO

        #self.dbrh = TODO
        #self.dbzh = TODO
        #self.dbnh = TODO

        #dx = TODO
        
        # dL/dn = dL/dh_t * (1 - z)
        dn_t = delta * (1 - self.z)
        
        # dL/dz = dL/dh_t * (self.hidden - n)
        dz_t = delta * (self.hidden - self.n) # <-- Fix 2
        
        # dL/dh_prev_t = dL/dh_t * z
        # This will be accumulated
        dh_prev_t = delta * self.z

        # Backprop through candidate state n = tanh(n_linear)
        dn_linear = self.h_act.backward(dn_t)

        # Backprop through update gate z = sigmoid(z_linear)
        dz_linear = self.z_act.backward(dz_t)
        
        # Backprop through n_linear = Wnx*x + bnx + r * (Wnh*self.hidden + bnh)
        
        # Gradients for n_linear parameters
        self.dWnx = np.outer(dn_linear, self.x)
        self.dbnx = dn_linear
        
        # Gradients for the reset-gated part
        n_h_part_linear = (self.Wnh @ self.hidden + self.bnh) # <-- Fix 2
        
        # dL/dr = dL/dn_linear * (Wnh*self.hidden + bnh)
        dr_t = dn_linear * n_h_part_linear # <-- Fix 2
        
        # dL/d(Wnh*self.hidden + bnh) = dL/dn_linear * r
        d_n_h_part = dn_linear * self.r
        
        self.dWnh = np.outer(d_n_h_part, self.hidden) # <-- Fix 2
        self.dbnh = d_n_h_part
        
        # Accumulate gradient for h_prev_t
        dh_prev_t += d_n_h_part @ self.Wnh
        
        # Backprop through reset gate r = sigmoid(r_linear)
        dr_linear = self.r_act.backward(dr_t)  # <--- ADD THIS LINE
        
        # Backprop through z_linear = Wzx*x + bzx + Wzh*self.hidden + bzh
        self.dWzx = np.outer(dz_linear, self.x)
        self.dbzx = dz_linear
        self.dWzh = np.outer(dz_linear, self.hidden) # <-- Fix 2
        self.dbzh = dz_linear
        
        # Accumulate gradients for h_prev_t and x
        dh_prev_t += dz_linear @ self.Wzh
        dx_z = dz_linear @ self.Wzx
        
        # Backprop through r_linear = Wrx*x + brx + Wrh*self.hidden + brh
        self.dWrx = np.outer(dr_linear, self.x)
        self.dbrx = dr_linear
        self.dWrh = np.outer(dr_linear, self.hidden) # <-- Fix 2
        self.dbrh = dr_linear

        # Accumulate gradients for h_prev_t and x
        dh_prev_t += dr_linear @ self.Wrh
        dx_r = dr_linear @ self.Wrx

        # Get gradient for x from n_linear
        dx_n = dn_linear @ self.Wnx

        # Total gradient w.r.t. x
        dx = dx_n + dx_z + dx_r

        return dx, dh_prev_t
