import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden, latent_dimension, act=F.tanh):
        super().__init__()
        self.act = act
        self.encoder_layer_0 = nn.Linear(784, hidden)
        self.encoder_layer_1 = nn.Linear(hidden, hidden)
        self.encoder_layer_2 = nn.Linear(hidden, hidden)
        self.encoder_layer_3 = nn.Linear(hidden, hidden)
        self.encoder_layer_4 = nn.Linear(hidden, hidden)
        self.encoder_layer_5 = nn.Linear(hidden, latent_dimension) # mean, var 합쳐서 24개

    def forward(self, x):
        x = self.act(self.encoder_layer_0(x))
        x = self.act(self.encoder_layer_1(x) + x)
        x = self.act(self.encoder_layer_2(x) + x)
        x = self.act(self.encoder_layer_3(x) + x)
        x = self.act(self.encoder_layer_4(x) + x)
        out = self.encoder_layer_5(x)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden, latent_dimension, class_dim = 0, act = F.tanh):
        super().__init__()
        self.act = act
        self.decoder_layer_0 = nn.Linear(latent_dimension//2 + class_dim, hidden) # mean 12개, var 12개가 z로 dimension 12개
        self.decoder_layer_1 = nn.Linear(hidden, hidden)
        self.decoder_layer_2 = nn.Linear(hidden, hidden)
        self.decoder_layer_3 = nn.Linear(hidden, hidden)
        self.decoder_layer_4 = nn.Linear(hidden, hidden)
        self.decoder_layer_5 = nn.Linear(hidden, 784)

    def forward(self, x):
        x = self.act(self.decoder_layer_0(x))
        x = self.act(self.decoder_layer_1(x) + x)
        x = self.act(self.decoder_layer_2(x) + x)
        x = self.act(self.decoder_layer_3(x) + x)
        x = self.act(self.decoder_layer_4(x) + x)
        out = F.sigmoid(self.decoder_layer_5(x))
        return out

class M1(nn.Module):
    def __init__(self, hidden, latent_dimension):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(hidden, latent_dimension)
        self.decoder = Decoder(hidden, latent_dimension)
        self.classifier = nn.Linear(latent_dimension, 10)
    def Sampling(self, mean, std):
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z
    def forward(self, x):
        # Encoder
        latent = self.encoder(x)
        y = F.softmax(self.classifier(latent), dim=1)
        mean, log_var = latent[:, 0:self.latent_dimension//2], latent[:, self.latent_dimension//2:]
        # decoder
        z = self.Sampling(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var, y

class M2(nn.Module):
    def __init__(self, hidden, latent_dimension, act = F.tanh):
        super().__init__()
        self.act = act
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(hidden, latent_dimension)
        self.decoder = Decoder(hidden, latent_dimension, class_dim=10)
        self.linear1 = nn.Linear(784+self.latent_dimension, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 10)

    def Sampling(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        z = mean + std * epsilon
        return z
    def forward(self, x, y):
        # Encoder
        latent = self.encoder(x)
        cat_latent = torch.cat((latent, x), dim=1)
        latent_1 = self.act(self.linear1(cat_latent))
        latent_2 = self.act(self.linear2(latent_1))
        rec_y = F.softmax(self.classifier(latent_2), dim=1)
        mean, log_var = latent[:, 0:self.latent_dimension//2], latent[:, self.latent_dimension//2:]
        # decoder
        z = self.Sampling(mean, log_var)
        if y==None: # Unlabel
            zy = torch.cat((z, rec_y), dim=1)
        else: # Label
            zy = torch.cat((z, y), dim=1)
        out = self.decoder(zy)
        return out, mean, log_var, rec_y
