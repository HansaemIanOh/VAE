import torch
import matplotlib.pyplot as plt

def display(data):
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu().detach()
            data = data.numpy()
    data = data.reshape(28, 28)
    plt.imshow(data, cmap='gray')
    plt.show()