#La description de cette classe est disponible dans le fichier Hybrid neural network.ipynbfrom qiskit import Aer
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from Net import Net
import time
import torch.nn as nn
from qiskit.providers.aer import AerSimulator

import warnings
warnings.filterwarnings("ignore")


simulator = AerSimulator()

# Concentrating on the first 100 samples
n_samples = 100
#----------------------------------------------
# Train dataset
#----------------------------------------------
X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))


X_train.data = X_train.data[:n_samples]
X_train.targets = X_train.targets[:n_samples]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

#----------------------------------------------
# Test dataset
#----------------------------------------------

n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

X_test.data = X_test.data[:n_samples]
X_test.targets = X_test.targets[:n_samples]
test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

#----------------------------------------------
# Training on GPU
#----------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ",device)

model = Net("GPU")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss().cuda()
loss_func = loss_func.to(device)
epochs = 20
loss_list = []
#Time evaluation
model_evaluation = []
loss_computing = []
backward_computing = []
optimizer_computing = []
quantum_circuit_computing = []


model.train()
start_global = time.time()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        # Forward pass
        start = time.time()
        output = model(data)
        output = output.to(device)
        stop = time.time()
        model_evaluation.append(stop-start)
        # Calculating loss
        start = time.time()
        loss = loss_func(output, target)
        # loss = loss.to(device)
        stop = time.time()
        loss_computing.append(stop-start)
        # Backward pass
        start = time.time()
        loss.backward()
        stop = time.time()
        backward_computing.append(stop-start)
        # Optimize the weights
        start = time.time()
        optimizer.step()
        stop = time.time()
        optimizer_computing.append(stop-start)
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))
stop_global = time.time()
print("Training on ",device, " took ",stop_global-start_global)
plt.rcParams['figure.figsize'] = [10, 5]
fig, (ax0,ax1) = plt.subplots(2,2)
ax0[0].plot(range(len(model_evaluation)),model_evaluation)
ax0[0].title.set_text("Model evalutation")
ax0[1].plot(range(len(loss_computing)),loss_computing)
ax0[1].title.set_text("Loss computing")
ax1[0].plot(range(len(backward_computing)),backward_computing)
ax1[0].title.set_text("Backward computing")
ax1[1].plot(range(len(optimizer_computing)),optimizer_computing)
ax1[1].title.set_text("Optimizer computing")
# plt.savefig("mploutput/timeCombinedCpu.png")
plt.clf()

plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Neg Log Likelihood Loss')
# plt.savefig("mploutput/LossTimeCpu.png")

#----------------------------------------------
# Scoring
#----------------------------------------------

model.eval()
with torch.no_grad():
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data,target = data.to(device), target.to(device)
        output = model(data)
        output = output.to(device)
        
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )




#-------------------------------------
# Training on CPU
#-------------------------------------
model = Net("CPU")
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()
epochs = 20
loss_list = []
#Time evaluation
model_evaluation = []
loss_computing = []
backward_computing = []
optimizer_computing = []
quantum_circuit_computing = []

device = torch.device('cpu')
model.train()
start_global = time.time()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        start = time.time()
        output = model(data)
        stop = time.time()
        model_evaluation.append(stop-start)
        # Calculating loss
        start = time.time()
        loss = loss_func(output, target)
        stop = time.time()
        loss_computing.append(stop-start)
        # Backward pass
        start = time.time()
        loss.backward()
        stop = time.time()
        backward_computing.append(stop-start)
        # Optimize the weights
        start = time.time()
        optimizer.step()
        stop = time.time()
        optimizer_computing.append(stop-start)
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))
stop_global = time.time()
print("Training on CPU took ",stop_global-start_global)