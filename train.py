import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
seed = 7
np.random.seed(seed)
torch.manual_seed(seed)

#random crop padding 4
#random horizontal reflection probability=50% 

train_transform = transforms.Compose([
             transforms.RandomCrop(32, padding=4)
            ,transforms.RandomHorizontalFlip(p=0.5)
            ,transforms.ToTensor()
           ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
            transforms.ToTensor()
            ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', 
                                         train=True, download=True, transform=train_transform)

test_set = torchvision.datasets.CIFAR10(root='./cifardata', 
                                        train=False, download=True, transform=test_transform)

test_set,valid_set = torch.utils.data.random_split(test_set,(5000,5000))

print(len(test_set),"  ",len(valid_set))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=64)#, sampler=train_sampler, num_workers=10)

test_loader = torch.utils.data.DataLoader(test_set, 
                                          batch_size=len(test_set))#, sampler=test_sampler, num_workers=10)

val_loader = torch.utils.data.DataLoader(valid_set, 
                                          batch_size=64)#, sampler=test_sampler, num_workers=10)

import torch.optim as optim

#sgd optimizer with adjustable learning_rate
def createOptimizer(model, learning_rate=0.1):   
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.9,weight_decay=0.0005)
    
    return optimizer

class SimpleCNN(torch.nn.Module):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv2d_11 = torch.nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2d_12 = torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv2d_21 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2d_22 = torch.nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)

        self.conv2d_31 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv2d_32 = torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv2d_33 = torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)

        self.conv2d_41 = torch.nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv2d_42 = torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.conv2d_51 = torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.Batchnorm_1=torch.nn.BatchNorm2d(64)
        self.Batchnorm_2=torch.nn.BatchNorm2d(128)
        self.Batchnorm_3=torch.nn.BatchNorm2d(256)
        self.Batchnorm_4=torch.nn.BatchNorm2d(512)

        self.dropout2d_1=torch.nn.Dropout2d(p=0.3)
        self.dropout2d_2=torch.nn.Dropout2d(p=0.4)
        self.dropout2d_3=torch.nn.Dropout2d(p=0.5)

        self.dropout1d=torch.nn.Dropout(p=0.5)
        
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.fc = torch.nn.Linear(512, 10)
                
    def forward(self, x):

        ############################# Phase 1
        #print(x.size())
        x = F.relu(self.conv2d_11(x))
        x = self.dropout2d_1(x) #rate =0.3
        x = self.Batchnorm_1(x) #input 64
        #print(x.size())
        
        x = F.relu(self.conv2d_12(x))
        x = self.dropout2d_1(x) #rate=0.3
        x = self.Batchnorm_1(x) #input 64
        #print(x.size())

        x = self.maxpool2d(x)
        #print(x.size())
        ############################# Phase 2
        x = F.relu(self.conv2d_21(x))
        x = self.dropout2d_1(x) #rate=0.3
        x = self.Batchnorm_2(x) #input 128
        #print(x.size())
        
        x = F.relu(self.conv2d_22(x))
        x = self.dropout2d_1(x) #rate=0.3
        x = self.Batchnorm_2(x) #input 128
        #print(x.size())
        
        x = self.maxpool2d(x)
        #print(x.size())
        ############################# Phase 3
        x = F.relu(self.conv2d_31(x))
        x = self.dropout2d_2(x) #rate=0.4
        x = self.Batchnorm_3(x) #input 256
        #print(x.size())
        
        x = F.relu(self.conv2d_32(x))
        x = self.dropout2d_2(x) #rate=0.4
        x = self.Batchnorm_3(x) #input 256
        #print(x.size())
        
        x = F.relu(self.conv2d_33(x))
        x = self.dropout2d_2(x) #rate=0.4
        x = self.Batchnorm_3(x) #input 256
        #print(x.size())
        
        x = self.maxpool2d(x)
        #print(x.size())
        ############################# Phase 4
        x = F.relu(self.conv2d_41(x))
        x = self.dropout2d_2(x)
        x = self.Batchnorm_4(x)
        #print(x.size())
        
        x = F.relu(self.conv2d_42(x))
        x = self.dropout2d_2(x)
        x = self.Batchnorm_4(x)
        #print(x.size())
        
        x = self.maxpool2d(x)
        #print(x.size())
        ############################# Phase 5
        x = F.relu(self.conv2d_51(x))
        x = self.dropout2d_3(x)
        x = self.Batchnorm_4(x)
        #print(x.size())
        
        x = self.avgpool2d(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.dropout1d(x)
        x = F.relu(self.fc(x))
        x = self.dropout1d(x)
        #print(x.size())
        x = F.softmax(x)
        ###############################
        
        return(x)

import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

def trainNet(model, batch_size, n_epochs, learning_rate):
    
    lr=learning_rate
    
    #Print all of the hyperparameters of the training iteration:
    print("======= HYPERPARAMETERS =======")
    print("Batch size=", batch_size)
    print("Epochs=", n_epochs)
    print("Base learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)
        
    #Time for printing
    training_start_time = time.time()
    
    #Loss function"
    loss = torch.nn.CrossEntropyLoss()
    optimizer = createOptimizer(model, lr)   
    
    scheduler = ReduceLROnPlateau(optimizer, 'min'
                                  ,patience=3,factor=0.9817
                                 ,verbose=True,)
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        #save the weightsevery 10 epochs
        if epoch % 10 == 0 :
            torch.save(model.state_dict(), 'model.ckpt')
      
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        total_train_acc = 0
        epoch_time = 0
        
        for i, data in enumerate(train_loader, 0):
             
            inputs, labels = data
            
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            score, predictions = torch.max(outputs.data, 1)
            acc = (labels==predictions).sum()
            total_train_acc += acc.item()
            
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d} % \t | train_loss: {:.3f} | train_acc:{}% | took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every
                        ,int(acc), time.time() - start_time))

                epoch_time += (time.time() - start_time)
                
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
            
                inputs,labels=Variable(inputs.to(device)),Variable(labels.to(device))
                model.eval()
                y_pred = model(inputs)
                total_val_loss  += loss(y_pred, labels)
                _, predictions = torch.max(y_pred.data, 1)
                acc = (labels==predictions).sum()
                total_val_acc += acc.item()
        
        
        scheduler.step(total_val_loss)
        
        print("-"*30)
        print("Train loss = {:.2f} | Train acc = {:.1f}% | Val loss={:.2f} | Val acc: {:.2f} | took: {:.2f}s".format(
            total_train_loss / len(train_loader),total_train_acc/ len(train_loader)
            ,total_val_loss/len(val_loader),total_val_acc/len(val_loader),epoch_time))
        print("="*60)
        
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

CNN = SimpleCNN().to(device)
CNN.eval()

trainNet(CNN, batch_size=64, n_epochs=250, learning_rate=0.1)
