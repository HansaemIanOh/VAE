from Optimization import *
from DataLoader import *
from utils import *
import torch
import argparse
import matplotlib.pyplot as plt
Loader = ClassDataLoader()
data = Loader.Load()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--epochs', type=int, default=10, help='Iteration to run [default: 10]')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--hidden_layers', type=int, default=100, help='The number of hidden layers')
parser.add_argument('--algorithm', type=str, default="M2", help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--latent_dim', type=int, default=28, help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--lr', type=float, default=0.0001, help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--num_label', type=int, default=100, help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')

args = parser.parse_args()
device = torch.device('cuda:' + '{}'.format(args.gpu))
torch.manual_seed(args.seed)

Trainer = ClassOptimization(model_type = args.algorithm, latent_dimension=args.latent_dim, device=device, lr=args.lr, batch_size=32, alpha=0.1, epochs=args.epochs, hidden=args.hidden_layers, num_label=args.num_label)

model_name = args.algorithm

model = Trainer.model
model = model.to(device)
load = os.path.join('Parameters/'+args.algorithm, model_name+".pth")

model.load_state_dict(torch.load(load))
os.makedirs('TrainingCurve/'+args.algorithm, exist_ok=True)
history_load = os.path.join('TrainingCurve/'+args.algorithm, model_name+".npy")
history = np.load(history_load)
# ===================
# Display data
# ===================
train_x, train_y, valid_x, valid_y, test_x, test_y = data
def N2T(data, data_type=torch.float32): return torch.tensor(data, device=device, dtype=data_type)
def D2H(data, num_classes=10): return torch.nn.functional.one_hot(torch.tensor(data, device=device), num_classes=num_classes)

x = N2T(train_x)
y = D2H(train_y)
if args.algorithm !='M1':
    # rec_x, mean, std, rec_y = model(x, y)
    rec_x, mean, std = model(x, y)
elif args.algorithm=='M1':
    rec_x, mean, std = model(x)
for _ in range(10):
    display(x[_])
    display(rec_x[_])
exit()
# ===================
# Training Curve Plot
# ===================
loss = history[:,0]
loss_valid = history[:,1]
epochs = np.arange(1, len(loss) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, loss_valid, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()

# ===================
# Model Test
# ===================
def N2T(data, data_type=torch.float32): return torch.tensor(data, device=device, dtype=data_type)
train_x, train_y, valid_x, valid_y, test_x, test_y = data
test_y_torch = torch.tensor(test_y, device=device)
test_y_hot = torch.nn.functional.one_hot(test_y_torch, num_classes=10)

rec_x, mean, log_var, rec_y = model(N2T(test_x), test_y_hot)
# logpy = torch.mean((rec_y*1.0 - test_y_hot*1.0)**2)
# logpx = torch.mean((rec_x - test_x)**2)
# KL_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim = 1), dim = 0)
# loss = logpy + logpx + 0.001 * KL_loss
# loss = logpy + logpx
# loss = logpy
# ==================== accuracy 탐색
_, predicted_labels = torch.max(rec_y, 1)
true_labels = torch.argmax(test_y_hot, dim=1)
correct_predictions = (predicted_labels == true_labels).sum().item()
accuracy = correct_predictions / test_y_hot.size(0)

print("Test Accuracy : ", accuracy)