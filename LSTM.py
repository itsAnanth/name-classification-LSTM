import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # input weights

        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)


        # hidden weights
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

        # final gate
        self.W_y = nn.Linear(hidden_size, output_size)



    def forward(self, x, h_prev, c_prev):


        i = F.sigmoid(self.W_i(x) + self.U_i(h_prev))

        f = F.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o = F.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c = F.tanh(self.W_c(x) + self.U_c(h_prev))

        c_t = (f * c_prev) + (i * c)
        h_t = o * F.tanh(c_t)

        logits = self.W_y(h_t)
        out = F.softmax(logits, dim=-1)
        return out, h_t, c_t
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def init_state(self):
        return torch.zeros(1, self.hidden_size)