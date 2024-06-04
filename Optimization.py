import time
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from Model import *

class ClassOptimization:
    def __init__(self, model_type, latent_dimension, device, lr, batch_size, alpha, epochs, hidden, num_label): # model_type, latent_dimension, device, lr, batch_size, alpha, hidden
        self.model_type = model_type
        if self.model_type =='M1':
            print("The model is M1")
            self.model = M1(hidden, latent_dimension, act=F.leaky_relu)
        elif self.model_type =='M2':
            print("The model is M2")
            self.model = M2(hidden, latent_dimension)
        elif self.model_type =='M0':
            print("The model is M0")
            self.model = M2(hidden, latent_dimension)
        elif self.model_type =='ConditionalM1':
            print("The model is conditional M1")
            self.model = ConditionalM1(hidden, latent_dimension)
        else:
            print("Warning!!! The model is not chosen accurately.")
            exit()
        self.device = device
        
        self.model.to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha
        self.epochs = epochs
        self.num_label = num_label
    def TrainM0(self, data):
        def N2T(data, data_type=torch.float32): return torch.tensor(data, device=self.device, dtype=data_type)
        def D2H(data, num_classes=10): return torch.nn.functional.one_hot(torch.tensor(data, device=self.device), num_classes=num_classes)

        train_x, train_y, valid_x, valid_y, test_x, test_y = data
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad = 'amsgrad')
        
        # Training
        start_time = time.time()
        self.model.train()
        best_model = copy.deepcopy(self.model)
        history = []

        print("best_model initialization.")
        # M0
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            # Label
            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:(i+self.batch_size), ...] # (batch, channel, width, height)
                train_y_batch = train_y[i:(i+self.batch_size), ...]
                optimizer.zero_grad()
                train_y_hot = D2H(train_y_batch)
                loss, accuracy = self.LossM0(N2T(train_x_batch), train_y_hot)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            accuracies.append(accuracy)
            losses = np.array(losses)
            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(np.array(accuracies))
            valid_y_torch = torch.tensor(valid_y, device = self.device)
            valid_y_hot = torch.nn.functional.one_hot(valid_y_torch, num_classes=10)
            loss_valid, valid_accuracy = self.LossM0(N2T(valid_x), valid_y_hot)
            loss_valid_comparison, valid_accuracy_comparison = self.LossM2VAEUnLabelComparison(N2T(valid_x), valid_y_hot, best_model)
            if valid_accuracy > valid_accuracy_comparison:
                best_model = copy.deepcopy(self.model)
                print("Model stored.")

            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f} Accuracy : {:.5f} Valid Accuracy : {:.5f}'.format(epoch + 1, self.epochs, epoch_train_time, avg_loss, avg_accuracy, valid_accuracy))
            # loss evolution
            loss_save = [avg_loss, loss_valid.item()]
            history.append(loss_save)
        return best_model, history
    
    def TrainM1(self, data):
        def N2T(data, data_type=torch.float32): return torch.tensor(data, device=self.device, dtype=data_type)
        def D2H(data, num_classes=10): return torch.nn.functional.one_hot(torch.tensor(data, device=self.device), num_classes=num_classes)
        train_x, train_y, valid_x, valid_y, test_x, test_y = data
        
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad = 'amsgrad')
        
        # Training
        start_time = time.time()
        self.model.train()
        history = []

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i+self.batch_size, ...] # (batch, channel, width, height)
                train_y_batch = train_y[i:i+self.batch_size, ...]
                optimizer.zero_grad()
                train_y_hot = D2H(train_y_batch)

                loss = self.LossM1VAE(N2T(train_x_batch))
                
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            losses = np.array(losses)
            avg_loss = np.mean(losses)

            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f}'.format(epoch + 1, self.epochs, epoch_train_time, avg_loss))
            # loss evolution
            loss_save = [loss.item()]
            history.append(loss_save)
        return self.model, history       

    def TrainConditionalM1(self, data):
        def N2T(data, data_type=torch.float32): return torch.tensor(data, device=self.device, dtype=data_type)
        def D2H(data, num_classes=10): return torch.nn.functional.one_hot(torch.tensor(data, device=self.device), num_classes=num_classes)
        train_x, train_y, valid_x, valid_y, test_x, test_y = data
        
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad = 'amsgrad')
        
        # Training
        start_time = time.time()
        self.model.train()
        history = []
        # VAE training
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i+self.batch_size, ...] # (batch, channel, width, height)
                train_y_batch = train_y[i:i+self.batch_size, ...]
                optimizer.zero_grad()
                train_y_hot = D2H(train_y_batch)

                loss = self.LossConditionalM1VAE(N2T(train_x_batch), train_y_hot)
                
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            losses = np.array(losses)
            avg_loss = np.mean(losses)

            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f}'.format(epoch + 1, self.epochs, epoch_train_time, avg_loss))
            # loss evolution
            loss_save = [loss.item()]
            history.append(loss_save)
        # Classifier training
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i+self.batch_size, ...] # (batch, channel, width, height)
                train_y_batch = train_y[i:i+self.batch_size, ...]
                optimizer.zero_grad()
                train_y_hot = D2H(train_y_batch)

                loss = self.LossConditionalM1VAE(N2T(train_x_batch), train_y_hot)
                
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            losses = np.array(losses)
            avg_loss = np.mean(losses)

            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f}'.format(epoch + 1, self.epochs, epoch_train_time, avg_loss))
            # loss evolution
            loss_save = [loss.item()]
            history.append(loss_save)
        return self.model, history


    def TrainM2(self, data):
        def N2T(data, data_type=torch.float32): return torch.tensor(data, device=self.device, dtype=data_type)
        def D2H(data, num_classes=10): return torch.nn.functional.one_hot(torch.tensor(data, device=self.device), num_classes=num_classes)
        train_x, train_y, valid_x, valid_y, test_x, test_y = data
        
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad = 'amsgrad')
        
        # Training
        start_time = time.time()
        self.model.train()
        best_model = copy.deepcopy(self.model)
        history = []

        index_total = np.arange(train_x.shape[0])
        index_label = np.random.choice(index_total, self.num_label, replace=False)

        train_x_label = train_x[index_label]
        train_y_label = train_y[index_label]
        # M2

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            

            for i in range(0, train_x_label.shape[0], self.batch_size):
                train_x_label_batch = train_x_label[i:i+self.batch_size, ...] # (batch, channel, width, height)
                train_y_label_batch = train_y_label[i:i+self.batch_size, ...]
                ratio = 50000 // self.num_label
                train_x_batch = train_x[ratio * i:ratio * (i+self.batch_size), ...] # (batch, channel, width, height)
                train_y_batch = train_y[ratio * i:ratio * (i+self.batch_size), ...]
                optimizer.zero_grad()
                train_y_label_hot = D2H(train_y_label_batch)
                train_y_hot = D2H(train_y_batch)

                loss_label, accuracy = self.LossM2VAELabel(N2T(train_x_label_batch), train_y_label_hot)
                loss_unlabel, _ = self.LossM2VAEUnLabel(N2T(train_x_batch), train_y_hot)
                loss = loss_label
                if epoch > 10:
                    alpha = 0.1
                    # for g in optimizer.param_groups:
                    #     g['lr'] = self.lr * 0.1
                else:
                    alpha = 0.01 # initial.
                
                
                loss = (1-alpha) * loss + alpha * loss_unlabel
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            accuracies.append(accuracy)
            losses = np.array(losses)
            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(np.array(accuracies))
            valid_y_torch = torch.tensor(valid_y, device = self.device)
            valid_y_hot = torch.nn.functional.one_hot(valid_y_torch, num_classes=10)
            loss_valid, valid_accuracy = self.LossM2VAEUnLabel(N2T(valid_x), valid_y_hot)
            loss_valid_comparison, valid_accuracy_comparison = self.LossM2VAEUnLabelComparison(N2T(valid_x), valid_y_hot, best_model)
            if valid_accuracy > valid_accuracy_comparison:
                best_model = copy.deepcopy(self.model)
                print("Model stored.")

            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f} Accuracy : {:.5f} Valid Accuracy : {:.5f}'.format(epoch + 1, self.epochs, epoch_train_time, avg_loss, avg_accuracy, valid_accuracy))
            # loss evolution
            loss_save = [loss.item(), loss_valid.item()]
            history.append(loss_save)
        return best_model, history       
    
    def LossM0(self, x, y):
        rec_x, mean, log_var, rec_y = self.model(x, y)
        loss = F.cross_entropy(rec_y.float(), y.float())
        # ==================== accuracy 탐색
        _, predicted_labels = torch.max(rec_y, 1)
        true_labels = torch.argmax(y, dim=1)
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        # ====================
        return loss, accuracy


    def LossM1VAE(self, x):
        rec_x, mean, log_var, y_hat = self.model(x)
        rec_loss = nn.functional.binary_cross_entropy(rec_x, x, reduction='sum')
        KL_loss = - 0.5 * torch.mean(1 + log_var - mean**2 - log_var.exp())
        loss = rec_loss + KL_loss
        return loss

    def LossM1C(self, x, y):
        rec_x, mean, log_var, y_hat = self.model(x)
        loss_y = nn.functional.binary_cross_entropy(y_hat, y, reduction='sum')
        loss = loss_y
        return loss

    def LossConditionalM1VAE(self, x, y):
        rec_x, mean, log_var = self.model(x, y)
        # rec_loss = torch.mean((rec_x - x)**2)
        rec_loss = nn.functional.binary_cross_entropy(rec_x, x, reduction='sum')
        KL_loss = - 0.5 * torch.mean(1 + log_var - mean**2 - log_var.exp())
        loss = rec_loss + KL_loss
        return loss

    def LossM1Cls(self, x, y):
        _, _, _, rec_y = self.model(x)
        loss = nn.functional.cross_entropy(rec_y, y*1.0)
        # Get the predicted labels (the index of the max log-probability)
        _, predicted_labels = torch.max(rec_y, 1)
        true_labels = torch.argmax(y, dim=1)
        # Calculate the number of correct predictions
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        return loss, accuracy

    def ValidLoss(self, x, y):
        _, _, _, rec_y = self.model(x)
        loss = nn.functional.cross_entropy(rec_y, y*1.0)
        # Get the predicted labels (the index of the max log-probability)
        _, predicted_labels = torch.max(rec_y, 1)
        true_labels = torch.argmax(y, dim=1)
        # Calculate the number of correct predictions
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        return loss, accuracy

    def LossM2VAELabel(self, x, y):
        rec_x, mean, log_var, rec_y = self.model(x, y)
        logpy = F.cross_entropy(rec_y.float(), y.float())
        logpx = F.binary_cross_entropy(rec_x, x)
        KL_loss = - 0.5 * torch.mean(1 + log_var - mean**2 - log_var.exp())
        loss = logpy + logpx + KL_loss
        # ==================== accuracy 탐색
        _, predicted_labels = torch.max(rec_y, 1)
        true_labels = torch.argmax(y, dim=1)
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        # ====================
        return loss, accuracy

    def LossM2VAEUnLabel(self, x, y):
        y_none = None
        rec_x, mean,log_var, rec_y = self.model(x, y_none)
        logpx = F.binary_cross_entropy(rec_x, x)
        max_indices = torch.argmax(rec_y, dim=1)
        # new_y = torch.nn.functional.one_hot(max_indices, num_classes=rec_y.shape[1])
        # logpy = F.cross_entropy(rec_y.float(), new_y.float())
        # logpy = torch.mean(torch.mul(rec_y.float(), torch.log(rec_y.float())))
        logpy = -torch.mean(torch.sum(rec_y * torch.log(rec_y + 1e-9), dim=1))
        KL_loss = - 0.5 * torch.mean(1 + log_var - mean**2 - log_var.exp())
        loss = logpy + logpx + KL_loss
        # ==================== accuracy 탐색
        _, predicted_labels = torch.max(rec_y, 1)
        # print(predicted_labels)
        true_labels = torch.argmax(y, dim=1)
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        # ====================
        return loss, accuracy

    def LossM2VAEUnLabelComparison(self, x, y, model):
        y_none = None
        rec_x, mean, std, rec_y = model(x, y_none)
        logpx = F.binary_cross_entropy(rec_x, x)
        entropy = - torch.mean(rec_y * torch.log(rec_y))
        KL_loss = torch.mean(- 0.5 * torch.sum(1 + torch.log(std**2) - mean ** 2 - std**2, dim = 1), dim = 0)
        loss = entropy + logpx + KL_loss
        # ==================== accuracy 탐색
        _, predicted_labels = torch.max(rec_y, 1)
        true_labels = torch.argmax(y, dim=1)
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        # ====================
        return loss, accuracy
