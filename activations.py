import torch
import math

def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x,3.0))))