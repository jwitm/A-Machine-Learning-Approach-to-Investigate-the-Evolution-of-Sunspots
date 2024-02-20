import os
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, confusion_matrix
from matplotlib.ticker import MultipleLocator
import wandb


class Trainer():
    def __init__(self, project_name: str, Model: torch.nn.Module, loss_func: torch.nn.Module,  directory: str, lr_cnn: float, lr_transformer: float, num_epochs: int, batch_size: int, optimizer: str = 'Adam', warmup_epochs: int = None):
        """
        Model: model architecture
        loss_func: which loss function you want to use nn.Module
        directory: base directory where everything should be saved as str
        lr: initial learning rate, if warmup is used, usually 0
        num_epochs: the number of epochs the base model should be trained
        batch_size: size of batch
        optimizer: str, choose optimizer from list, default Adam
        schedule: which scheduler should be used (fine tuning is necessary)
        warmup_epochs: how many warmup epochs should be used
        """
        
        pretrained_filename_single = os.path.join(f"{directory}/lr_cnn_{lr_cnn}_lr_trans_{lr_transformer}_ep{num_epochs}.tar")  # this is the directory, where the model should be saved

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # use GPU if available

        self.directory = directory  # directory where the model should be saved
        self.num_epochs = num_epochs # maximum number of epochs to train 
        self.lr_cnn = lr_cnn # learning rate for the cnn
        self.lr_transformer = lr_transformer # learning rate for the transformer
        self.batch_size = batch_size
        self.loss = loss_func
        self.Model = Model
        self.project_name = project_name # name of the project used in weights and biases

        self.already_trained_single = None  

        self.weight_decay = 5e-1  # weight decay for the optimizer
        self.accumulation_steps = 64   # number of steps, where the gradients are accumulated
        self.early_stopping = 50    # number of epochs, where the loss does not decrease to stop training
        self.l1_lambda = 0.01   # L1 regularization strength
        self.l2_lambda = 0.01   # L2 regularization strength
        
        if os.path.isfile(pretrained_filename_single): # check if model already exists
            print("single model already exists, loading...\n")
            model = Model   # set architecture for model
            state_dict = torch.load(pretrained_filename_single, map_location=torch.device(self.device)) # get parameters for trained model
            model.load_state_dict(state_dict)    # load model with parameter
            self.model = model.to('cpu') # push model to cpu for memory reasons
            self.already_trained_single = True
                
        else:
            self.already_trained_single = False

            if not os.path.exists(directory): # create base directory if not already existing
                os.makedirs(directory)

            self.model = Model.to('cpu') # take the architecture and push model to cpu for memory reasons
            
            self.warmup_epochs = warmup_epochs  # number of epochs should be an istance of the class
            

            if optimizer == 'Adam': # loading the right optimizer depending on keyword
                self.optimizer_cnn = optim.Adam(self.model.convnet.parameters(), lr=lr_cnn, weight_decay=self.weight_decay)
                self.optimizer_transformer = optim.Adam(self.model.transformer.parameters(), lr=lr_transformer, weight_decay=self.weight_decay)
            elif optimizer == 'SGD':
                self.optimizer_cnn = optim.SGD(self.model.convnet.parameters(), lr=lr_cnn, momentum = 0.9, weight_decay=self.weight_decay)
                self.optimizer_transformer = optim.SGD(self.model.transformer.parameters(), lr=lr_transformer, momentum = 0.9, weight_decay=self.weight_decay)
            elif optimizer == 'RMSPROP':
                self.optimizer_cnn = optim.RMSprop(self.model.convnet.parameters(), lr=lr_cnn, weight_decay=self.weight_decay)
                self.optimizer_transformer = optim.RMSprop(self.model.transformer.parameters(), lr=lr_transformer, weight_decay=self.weight_decay)
            elif optimizer == 'AdamW':
                self.optimizer_cnn = optim.AdamW(self.model.convnet.parameters(), lr=lr_cnn, weight_decay=self.weight_decay)
                self.optimizer_transformer = optim.AdamW(self.model.transformer.parameters(), lr=lr_transformer, weight_decay=self.weight_decay)
            else:
                raise ValueError(f"{optimizer} is not built in, choose between \'Adam\', \'SGD\', \'RMSPROP\' and \'AdamW\'.")

            # set the learning rate schedulers
            self.scheduler_transformer_warmup = LambdaLR(self.optimizer_transformer, lr_lambda=lambda epoch: warmup_schedule(epoch, warmup_epochs), verbose = True) 
            self.scheduler_cnn_warmup = LambdaLR(self.optimizer_cnn, lr_lambda=lambda epoch: warmup_schedule(epoch, warmup_epochs), verbose = True) 
            self.scheduler_transformer = ReduceLROnPlateau(self.optimizer_transformer, mode = 'min', factor = 0.5, patience = 10, verbose = True)
            self.scheduler_cnn = ReduceLROnPlateau(self.optimizer_cnn, mode = 'min', factor = 0.5, patience = 10, verbose = True)


    def training(self, train_data, test_data):
        """
        This function trains or loads the model type choosen in the __init__()

        train_data: training data as a dataloader
        test_data: testing data as a dataloader
        """

        if self.already_trained_single == True:                                                         # return loaded base model
            print('returning \'base\' model...')
            return self.model, self.directory, self.num_epochs  
        
        else:
            print('Now training the base model...\n')
            self.model, self.directory, self.num_epochs = self.training_single(self,train_data, test_data)   # train the base model
            return self.model, self.directory, self.num_epochs

    @staticmethod
    def training_single(self, train_data, test_data):
        """
        This function is going to train the base model.
        """
        self.model = self.model.to(self.device)                                                             # push the model to the device

        best_epoch = 0
        best_loss = float("inf")            

        wandb.init(project=f"{self.project_name}", entity="jaenu99",
            config={"lr0 transformer": self.lr_transformer,
                    "lr0 cnn": self.lr_cnn,
                    "Model": self.model,
                    "epochs": self.num_epochs,
                    "accumulation steps": self.accumulation_steps,
                    "weight decay": self.weight_decay,
                    "early stopping": self.early_stopping,
                    "warumup epochs": self.warmup_epochs,
                    "optimizer CNN": self.optimizer_cnn,
                    "optimizer Transformer": self.optimizer_transformer
            })

        # generating empty torch tensor to save train- and test loss
        loss_list_train = torch.tensor([], dtype=torch.float32, device=self.device)                         # save all the training losses
        loss_list_test = torch.tensor([], dtype=torch.float32, device=self.device)                          # save all the test losses

        for epoch in tqdm(range(self.num_epochs)):                                                          # iterate through the number of epochs set in the __init__() func.

            for i, (data, target) in enumerate(tqdm(train_data,desc="base training loop")):
                self.model.train()  # set the model in training mode
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)  # calculate forward pass
                Loss = self.loss(outputs, target) # calculate loss
                loss_list_train = torch.cat((loss_list_train, torch.unsqueeze(Loss.detach(),0)), dim = 0)   # append the loss to the list, and rip it of the gradient

                Loss.backward() # calculate the gradients
                                        
                if (i+1)% self.accumulation_steps == 0: # update the weights every accumulation_steps
                    self.optimizer_cnn.step()
                    self.optimizer_transformer.step()
                    self.optimizer_cnn.zero_grad()
                    self.optimizer_transformer.zero_grad()
                del(outputs, Loss)

            self.model.eval()                                                                               # put the model in eval mode
            for i, (test, test_target) in enumerate(tqdm(test_data, desc="base testing loop")): 
                with torch.no_grad():
                    outputs_test = self.model(test.to(self.device)) # perform forward pass
                    loss_test = self.loss(outputs_test, test_target.to(self.device)) # calculate loss of training data
                    loss_list_test = torch.cat((loss_list_test, torch.unsqueeze(loss_test.detach(),0)),dim = 0) # append the loss to the list, and rip it of the gradient
                    del(outputs_test, loss_test)

            loss_train_average = torch.mean(loss_list_train[epoch*len(train_data):(1+epoch)*len(train_data)])   # calculate the average train for an epoch
            loss_test_average = torch.mean(loss_list_test[epoch*len(test_data):(1+epoch)*len(test_data)])     # calculate the average train for an epoch


            if epoch < self.warmup_epochs: # update warmup scheduler
                self.scheduler_transformer_warmup.step()
                self.scheduler_cnn_warmup.step()
            else:   # update normal scheduler
                self.scheduler_transformer.step(loss_test_average)
                self.scheduler_cnn.step(loss_test_average)

            if loss_test_average < best_loss: # save the model, if the loss does not increase for p more epochs
                best_model_state_dict = self.model.state_dict()
                best_epoch = epoch+1
                best_loss = loss_test_average
                print(f"\nbest epoch = {best_epoch}\n")
                p = 0
            else:
                p +=1

            if p == self.early_stopping:
                break

            accuracy, tss = self.specs(model = None, test_data = test_data, verbose = False)    # calculate specs

            wandb.log({"training loss": loss_train_average,
                       "testing loss": loss_test_average,
                       "TSS": tss,
                       "Accuracy": accuracy,
                       "lr_CNN:": self.optimizer_cnn.param_groups[0]['lr'],
                       "lr_Transformer:": self.optimizer_transformer.param_groups[0]['lr'],
                       "best epoch": best_epoch})
            print("\n")

        torch.save(best_model_state_dict,f"{self.directory}/lr_cnn_{self.lr_cnn}_lr_trans_{self.lr_transformer}_ep{best_epoch}.tar")              # save the model

        loss_list_train = loss_list_train.to("cpu")
        loss_list_test = loss_list_test.to("cpu")
        loss_list_train = loss_list_train.detach().numpy()
        loss_list_test = loss_list_test.detach().numpy()

        new_path = os.path.join(f"{'/'.join(self.directory.split('/')[:-1])}/lr_cnn_{self.lr_cnn}_lr_trans_{self.lr_transformer}_ep{best_epoch}")
    
        if (best_epoch != self.num_epochs and not os.path.exists(new_path)):
            os.rename(self.directory, new_path)
            self.directory = new_path

        filename = f"{self.directory}/base_loss.txt"
        with open(filename, "w") as file:
            max_length = max(len(loss_list_train), len(loss_list_test))
        
            for i in range(max_length):
                train_loss = loss_list_train[i] if i < len(loss_list_train) else -1
                test_loss = loss_list_test[i] if i < len(loss_list_test) else -1
                file.write(f"{train_loss}   {test_loss}\n")

        self.num_epochs = best_epoch

        self.model.load_state_dict(best_model_state_dict) 

        accuracy, tss = self.specs(model = None, test_data = test_data, verbose = False)
        
        wandb.log({"TSS": tss,
                   "Accuracy": accuracy,
                   "best epoch": self.num_epochs})
        
        wandb.finish()

        self.model = self.model.to('cpu')                                                                                       # push model to cpu (memory)


        print('the base model is now trained successfully!\n')
                
        return self.model, self.directory, self.num_epochs 

    
    def plot_loss(self,mode = 'epoch'):
        filename = os.path.join(f"{self.directory}/base_loss.txt")
        loaded_data = np.loadtxt(filename)
        loss_list_train = loaded_data[:, 0]
        loss_list_test = loaded_data[:, 1]

        loss_list_train = loss_list_train[loss_list_train >= 0]

        len_data = len(loss_list_train)/(self.num_epochs+self.early_stopping )

        loss_list_test = loss_list_test[loss_list_test >= 0]

        len_test = len(loss_list_test)/(self.num_epochs+self.early_stopping )

        if mode == 'normal':
            fig, ax1 = plt.subplots()
            ax1.plot(np.arange(1,len(loss_list_train)+1, 1)/self.batch_size, loss_list_train, label = "train loss", color = 'tab:blue')
            ax1.set_xlabel('number of batches training', color='tab:blue')
            ax1.tick_params(axis='x', labelcolor='tab:blue')

            ax2 = ax1.twiny()

            ax2.plot(np.arange(1,len(loss_list_test)+1, 1)/self.batch_size, loss_list_test, label = "test loss", color = 'tab:red')
            ax2.set_xlabel('number of batches testing',  color='tab:red')
            ax2.tick_params(axis='x', labelcolor='tab:red')
            ax1.set_ylabel(r"$\mathcal{L}$", rotation = 0)
            ax1.yaxis.grid(True)
            lines_train, labels_train = ax1.get_legend_handles_labels()
            lines_test, labels_test = ax2.get_legend_handles_labels()
            ax1.legend(lines_train + lines_test, labels_train + labels_test, loc='upper left')

            ax1.xaxis.set_major_locator(MultipleLocator(2*len_data))
            ax1.xaxis.set_minor_locator(MultipleLocator(1*len_data))

            ax2.xaxis.set_major_locator(MultipleLocator(2*len_test))
            ax2.xaxis.set_minor_locator(MultipleLocator(1*len_test))

            ax1.grid(True)

            plt.tight_layout()
            plt.savefig(f'{self.directory}/loss.png', dpi = 1200)
            plt.close()

        elif mode == 'epoch':
            window_size_train = int(len(loss_list_train)/(self.num_epochs+self.early_stopping ))
            loss_list_train_ave = np.convolve(loss_list_train, np.ones(window_size_train)/window_size_train, mode='valid')

            window_size_test = int(len(loss_list_test)/(self.num_epochs+self.early_stopping ))
            
            print('window_size train:', window_size_train)
            print('window_size test:', window_size_test)

            loss_list_test_ave = np.convolve(loss_list_test, np.ones(window_size_test)/window_size_test, mode='valid')

            fig, ax1 = plt.subplots()
            ax1.plot(np.arange(0,len(loss_list_train_ave), 1)/window_size_train+1, loss_list_train_ave, label = "train loss", color = 'tab:blue')

            ax1.plot(np.arange(0,len(loss_list_test_ave), 1)/window_size_test+1, loss_list_test_ave, label = "test loss", color = 'red')
            ax1.set_xlabel('number of epochs')

            if self.num_epochs>=8:
                ax1.xaxis.set_major_locator(MultipleLocator(self.num_epochs//4))
                ax1.xaxis.set_minor_locator(MultipleLocator(self.num_epochs//8))
            else:
                ax1.xaxis.set_major_locator(MultipleLocator(2))
                ax1.xaxis.set_minor_locator(MultipleLocator(1))

            ax1.set_ylabel(r"$\mathcal{L}$", rotation = 0)
            ax1.axvline(x=self.num_epochs, color='tab:olive', linestyle='--', label=r'best model')
            ax1.grid(True)

            ax1.legend()

            plt.tight_layout()
            plt.savefig(f'{self.directory}/loss.png', dpi = 1200)
            plt.close()


    @staticmethod
    def find_threshhold(self, model,test_data):
        predicted_probs = torch.tensor([], dtype=torch.float32, device=self.device) 
        test_labels = torch.tensor([], dtype=torch.float32, device=self.device) 
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for test, test_label in tqdm(test_data, desc="determing threshold"):
                test = test.to(self.device)
                test_label = test_label.to(self.device)
                output = model(test)
                predicted_probs = torch.cat((predicted_probs, output.flatten()), dim = 0)
                test_labels = torch.cat((test_labels, test_label.flatten()), dim = 0)

        predicted_probs = predicted_probs.cpu().numpy()  # Transfer to CPU and then convert to numpy
        test_labels = test_labels.cpu().numpy()
        
        precision, recall, thresholds = precision_recall_curve(test_labels, predicted_probs)

        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        predicted_labels = (predicted_probs >= optimal_threshold).astype(int)

        return predicted_labels, test_labels #np.max(accuracies)

    def specs(self,model, test_data, verbose = True):
        if model == None:
            model = self.model
        if isinstance(model, list):
            acc = []
            TSS = []
            for x in tqdm(model, desc="calculating specs"):
                predicted_labels, test_labels = self.find_threshhold(self,x, test_data)

                conf_matrix = confusion_matrix(test_labels, predicted_labels)
                tn, fp, fn, tp = conf_matrix.ravel()

                accuracy = (tp+tn)/(tp+tn+fp+fn)
                sensitivity = tp/(tp + fn)
                specificity = tn / (tn + fp)
                tss = sensitivity + specificity-1

                if verbose == True:
                    print(f'accuracy = {accuracy}')
                    print(f"TSS = {tss}")

                acc.append(accuracy)
                TSS.append(tss)
            return acc, TSS

        else:
            predicted_labels, test_labels = self.find_threshhold(self,model, test_data)

            conf_matrix = confusion_matrix(test_labels, predicted_labels)
            tn, fp, fn, tp = conf_matrix.ravel()

            acc = (tp+tn)/(tp+tn+fp+fn)
            sensitivity = tp/(tp + fn)
            specificity = tn / (tn + fp)
            TSS = sensitivity + specificity-1

            if verbose:
                print(f'accuracy = {acc}')
                print(f"TSS = {TSS}")

            return acc, TSS
  
def warmup_schedule(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 1