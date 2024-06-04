from Optimization import *
from DataLoader import *
import torch
import argparse

Loader = ClassDataLoader()
data = Loader.Load()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--epochs', type=int, default=10, help='Iteration to run [default: 10]')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--hidden_layers', type=int, default=100, help='The number of hidden layers')
parser.add_argument('--algorithm', type=str, default="M2", help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--latent_dim', type=int, default=28, help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--lr', type=float, default=0.00001, help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--batch_size', type=int, default=10, help='Type the algorithm name : Pinn or ConvPinn or ConvPinnCat')
parser.add_argument('--num_label', type=int, default=100, help='')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:' + '{}'.format(args.gpu))
torch.manual_seed(args.seed)

Trainer = ClassOptimization(model_type = args.algorithm, latent_dimension=args.latent_dim, device=device, lr=args.lr, batch_size=args.batch_size, alpha=0.1, epochs=args.epochs, hidden=args.hidden_layers, num_label=args.num_label)

model_name = args.algorithm
if args.algorithm=='M0':
    print("This is M0")
    model, history = Trainer.TrainM0(data)
elif args.algorithm=='M1':
    print("This is M1")
    save = os.path.join('Parameters/'+args.algorithm, model_name+".pth")
    model = Trainer.model
    # model.load_state_dict(torch.load(save))
    # model, history = Trainer.TrainM1(data, model)
    model, history = Trainer.TrainM1(data, 0)
elif args.algorithm=='M2':
    print("This is M2")
    model, history = Trainer.TrainM2(data)
elif args.algorithm=='ConditionalM1':
    print("This is conditional M1")
    model, history = Trainer.TrainConditionalM1(data)
else:
    print("Wrong model is chosen.")
    exit()
os.makedirs('Parameters/'+args.algorithm, exist_ok=True)
save = os.path.join('Parameters/'+args.algorithm, model_name+".pth")
torch.save(model.state_dict(), save)
os.makedirs('TrainingCurve/'+args.algorithm, exist_ok=True)
history_save = os.path.join('TrainingCurve/'+args.algorithm, model_name+".npy")
history = np.array(history)
np.save(history_save, history)

