import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, eword_size):
        super(Highway, self).__init__()

        self.eword_size = eword_size
        self.w_proj = nn.Linear(self.eword_size, self.eword_size, bias=True)
        # self.b_proj = None
        self.w_gate = nn.Linear(self.eword_size, self.eword_size, bias=True)
        # self.b_gate = None
        self.highway_ReLU = nn.ReLU()

    def forward(self, x_conv: torch.Tensor):
        x_proj_pre = self.w_proj(x_conv)
        x_proj = self.highway_ReLU(x_proj_pre)
        x_gate_pre = self.w_gate(x_proj)
        x_gate = torch.sigmoid(x_gate_pre)

        x_highway = x_gate*x_proj + (1-x_gate)*x_conv
        return x_highway