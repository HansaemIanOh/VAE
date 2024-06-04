import torch
import torch.nn as nn
import torch.nn.functional as F
# =-=-=-=-
# Classifier Input : x, Output : latent2
# =-=-=-=-
class Classifier(nn.Module):
    def __init__(self, hidden, act=F.tanh):
        super().__init__()
        self.act = act
        self.encoder_layer_0 = nn.Linear(784, hidden)
        self.encoder_layer_1 = nn.Linear(hidden, hidden)
        self.encoder_layer_2 = nn.Linear(hidden, hidden)
        self.encoder_layer_3 = nn.Linear(hidden, hidden)
        self.encoder_layer_4 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = self.act(self.encoder_layer_0(x))
        x = self.act(self.encoder_layer_1(x))
        x = self.act(self.encoder_layer_2(x))
        x = self.act(self.encoder_layer_3(x))
        out = self.act(self.encoder_layer_4(x))
        return out
# =-=-=-=-
# Encoder Input : x, Output : latent2
# =-=-=-=-
class Encoder(nn.Module):
    def __init__(self, hidden, act=F.tanh):
        super().__init__()
        self.act = act
        self.encoder_layer_0 = nn.Linear(784, hidden)
        self.encoder_layer_1 = nn.Linear(hidden, hidden)
        self.encoder_layer_2 = nn.Linear(hidden, hidden)
        self.encoder_layer_3 = nn.Linear(hidden, hidden)
        self.encoder_layer_4 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = self.act(self.encoder_layer_0(x))
        x = self.act(self.encoder_layer_1(x))
        x = self.act(self.encoder_layer_2(x))
        x = self.act(self.encoder_layer_3(x))
        out = self.act(self.encoder_layer_4(x))
        return out
# =-=-=-=-
# Decoder Input : zy, Output : x
# =-=-=-=-
class Decoder(nn.Module):
    def __init__(self, hidden, act = F.tanh):
        super().__init__()
        self.act = act
        self.decoder_layer_1 = nn.Linear(hidden, hidden)
        self.decoder_layer_2 = nn.Linear(hidden, hidden)
        self.decoder_layer_3 = nn.Linear(hidden, hidden)
        self.decoder_layer_4 = nn.Linear(hidden, hidden)
        self.decoder_layer_5 = nn.Linear(hidden, 784)

    def forward(self, x):
        x = self.act(self.decoder_layer_1(x))
        x = self.act(self.decoder_layer_2(x))
        x = self.act(self.decoder_layer_3(x))
        x = self.act(self.decoder_layer_4(x))
        out = F.sigmoid(self.decoder_layer_5(x))
        return out
# =-=-=-=-
# M1 conventional vae
# =-=-=-=-
class M1(nn.Module):
    def __init__(self, hidden, latent_dimension, act = F.tanh):
        super().__init__()
        self.act = act
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(hidden)
        self.decoder = Decoder(hidden)
        self.classifier = Classifier(hidden)
        # mean layer
        self.layer_mean = nn.Linear(hidden, self.latent_dimension)
        self.layer_var = nn.Linear(hidden, self.latent_dimension)
        self.layer_y = nn.Linear(2 * hidden, 10)
        # Combine z and y to hidden layer
        self.layer_latent2 = nn.Linear(self.latent_dimension, hidden)
    def Sampling(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        z = mean + std * epsilon
        return z
    def forward(self, x):
        latent1 = self.encoder(x)
        mean = self.layer_mean(latent1)
        log_var = self.layer_var(latent1)
        z = self.Sampling(mean, log_var)
        latent2 = self.act(self.layer_latent2(z))
        x_hat = self.decoder(latent2)
        # ====================
        # Classifier network
        # ====================
        latent_y = self.classifier(x)
        latent2 = torch.cat((latent1, latent_y), dim=1)
        # latent2 = torch.cat((z, latent_y))
        y_hat = self.act(self.layer_y(latent2))
        return x_hat, mean, log_var, y_hat
# =-=-=-=-
# M1 conditional vae
# =-=-=-=-
class ConditionalM1(nn.Module):
    def __init__(self, hidden, latent_dimension, act = F.tanh):
        super().__init__()
        self.act = act
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(hidden)
        self.decoder = Decoder(hidden)
        self.classifier = Classifier(hidden)
        # latent layer
        self.layer_mean = nn.Linear(hidden, self.latent_dimension)
        self.layer_var = nn.Linear(hidden, self.latent_dimension)
        self.layer_y = nn.Linear(2 * hidden, 10)
        # Combine z and y to hidden layer
        self.layer_latent2 = nn.Linear(self.latent_dimension + 10, hidden)
        
        self.embedding = nn.Linear(10, hidden)
        self.embedding_latent = nn.Linear(self.latent_dimension + hidden, hidden)
    def Sampling(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        z = mean + std * epsilon
        return z
    def forward(self, x, y):
        # Encoder
        latent1 = self.encoder(x)
        # latent1 -> mean
        # latent1 -> log var
        mean = self.act(self.layer_mean(latent1))
        log_var = self.act(self.layer_var(latent1))
        z = self.Sampling(mean, log_var)
        # zy = torch.cat((z, y), dim=1)
        # embedding of y
        ey = self.act(self.embedding(y.float()))
        zy = torch.cat((z, ey), dim=1)
        latent2 = self.act(self.embedding_latent(zy))
        # embedding of y
        # latent2 = self.act(self.layer_latent2(zy))
        out = self.decoder(latent2)
        return out, mean, log_var

# M2 version1 y가 너무 encoder에 의존하기 때문에 성능이 떨어짐.
class M2(nn.Module):
    def __init__(self, hidden, latent_dimension, act = F.tanh):
        super().__init__()
        self.act = act
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(hidden)
        self.decoder = Decoder(hidden)
        self.classifier = Classifier(hidden)
        # latent layer
        self.layer_mean = nn.Linear(hidden, self.latent_dimension)
        self.layer_var = nn.Linear(hidden, self.latent_dimension)
        self.layer_y = nn.Linear(hidden, 10)
        # Combine z and y to hidden layer
        self.layer_latent2 = nn.Linear(self.latent_dimension + 10, hidden)
    def Sampling(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        z = mean + std * epsilon
        return z
    def forward(self, x, y):
        # Encoder
        latent1 = self.encoder(x)
        # latent1 -> mean
        # latent1 -> log var
        mean = self.act(self.layer_mean(latent1))
        log_var = self.act(self.layer_var(latent1))
        # latent2 -> y
        rec_y = F.softmax(self.layer_y(latent1), dim=1)
        # decoder
        z = self.Sampling(mean, log_var)
        if y==None: # Unlabel
            zy = torch.cat((z, rec_y), dim=1)
        else: # Label
            zy = torch.cat((z, y), dim=1)
        latent2 = self.act(self.layer_latent2(zy))
        out = self.decoder(latent2)
        return out, mean, log_var, rec_y
    

# M2 version2
# class M2(nn.Module):
#     def __init__(self, hidden, latent_dimension, act = F.tanh):
#         super().__init__()
#         self.act = act
#         self.latent_dimension = latent_dimension
#         self.encoder = Encoder(hidden)
#         self.decoder = Decoder(hidden)
#         self.classifier = Classifier(hidden)
#         # latent layer
#         self.layer_mean = nn.Linear(hidden, self.latent_dimension)
#         self.layer_var = nn.Linear(hidden, self.latent_dimension)
#         self.layer_y = nn.Linear(2 * hidden, 10)
#         # Combine z and y to hidden layer
#         self.layer_latent2 = nn.Linear(self.latent_dimension + 10, hidden)
#     def Sampling(self, mean, log_var):
#         epsilon = torch.randn_like(log_var)
#         std = torch.exp(0.5 * log_var)
#         z = mean + std * epsilon
#         return z
#     def forward(self, x, y):
#         # Encoder
#         latent1 = self.encoder(x)
#         # latent1 -> mean
#         # latent1 -> log var
#         mean = self.act(self.layer_mean(latent1))
#         log_var = self.act(self.layer_var(latent1))
#         # latent2 -> y
#         latent2 = self.classifier(x)
#         latent2 = torch.cat((latent1, latent2), dim=1)
#         rec_y = F.softmax(self.layer_y(latent2), dim=1)
#         # decoder
#         z = self.Sampling(mean, log_var)
#         if y==None: # Unlabel
#             zy = torch.cat((z, rec_y), dim=1)
#         else: # Label
#             zy = torch.cat((z, y), dim=1)
#         latent3 = self.act(self.layer_latent2(zy))
#         x_hat = self.decoder(latent3)
#         return x_hat, mean, log_var, rec_y
    
# M2 version3
# class M2(nn.Module):
#     def __init__(self, hidden, latent_dimension, act = F.tanh):
#         super().__init__()
#         self.act = act
#         self.latent_dimension = latent_dimension
#         self.encoder = Encoder(hidden)
#         self.decoder = Decoder(hidden)
#         self.classifier = Classifier(hidden)
#         # latent layer
#         self.layer_mean = nn.Linear(hidden, self.latent_dimension)
#         self.layer_var = nn.Linear(hidden, self.latent_dimension)
#         self.layer_y = nn.Linear(hidden, 10)
#         # Combine z and y to hidden layer
#         self.layer_latent2 = nn.Linear(self.latent_dimension + 10, hidden)
#     def Sampling(self, mean, log_var):
#         epsilon = torch.randn_like(log_var)
#         std = torch.exp(0.5 * log_var)
#         z = mean + std * epsilon
#         return z
#     def forward(self, x, y):
#         # Encoder
#         latent1 = self.encoder(x)
#         # latent1 -> mean
#         # latent1 -> log var
#         mean = self.act(self.layer_mean(latent1))
#         log_var = self.act(self.layer_var(latent1))
#         # latent2 -> y
#         latent2 = self.classifier(x)
#         rec_y = F.softmax(self.layer_y(latent2), dim=1)
#         # decoder
#         z = self.Sampling(mean, log_var)
#         if y==None: # Unlabel
#             zy = torch.cat((z, rec_y), dim=1)
#         else: # Label
#             zy = torch.cat((z, y), dim=1)
#         latent3 = self.act(self.layer_latent2(zy))
#         x_hat = self.decoder(latent3)
#         return x_hat, mean, log_var, rec_y
    
