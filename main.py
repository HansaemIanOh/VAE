import os
# os.system("python Train.py --algorithm M0")
# os.system("python Test.py --algorithm M0")

# os.system("python Train.py --algorithm M2 --epochs 100 --batch_size 10 --num_label 5000")
# os.system("python Test.py --algorithm M2 --epochs 100")

# os.system("python Train.py --algorithm M1 --epochs 100 --batch_size 100")
# os.system("python Test.py --algorithm M1 --epochs 100")

# os.system("python Train.py --algorithm ConditionalM1 --epochs 100 --batch_size 100")
# os.system("python Test.py --algorithm ConditionalM1 --epochs 100")

# os.system("python Train.py --algorithm M1 --epochs 300 --batch_size 100 --hidden_layers 400 --latent_dim 200 --lr 1e-4")
# os.system("python Test.py --algorithm M1 --epochs 300 --hidden_layers 400 --latent_dim 200")

# os.system("python Train.py --algorithm ConditionalM1 --epochs 300 --batch_size 100 --hidden_layers 400 --latent_dim 200 --lr 1e-4")
# os.system("python Test.py --algorithm ConditionalM1 --epochs 300 --hidden_layers 400 --latent_dim 200")

# os.system("python Train.py --algorithm M0 --epochs 30 --batch_size 100 --hidden_layers 400 --latent_dim 200 --lr 1e-4")

os.system("python Train.py --algorithm M2 --epochs 1000 --batch_size 100 --hidden_layers 400 --latent_dim 200 --num_label 1000 --lr 1e-4")