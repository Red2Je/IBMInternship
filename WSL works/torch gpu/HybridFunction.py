#La description de cette classe est disponible dans le fichier Hybrid neural network.ipynb
import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
from QuantumTraining import QuantumCircuitBuilder


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        if ctx.quantum_circuit.device == "GPU":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            grad_output = grad_output.to(device)
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        if ctx.quantum_circuit.device == "GPU":
            gradient_output = torch.tensor([gradient]).float().to(device)
        else:
             gradient_output = torch.tensor([gradient]).float()
        return gradient_output * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift, device):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuitBuilder(1, backend, shots, device)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)