#Importing issential Libraries
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torchvision.models as models
from torch.optim import lr_scheduler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch



def hello_img():
    print("Hello from image_classification.py!")


#Function to display Images
def image_display(image):        
  img_disp = image.clone()
  img_disp = img_disp.cpu()
  plt.figure(11)
  plt.imshow(np.array(img_disp))
  plt.show()


def sample_image_disp(images, labels):
  image_ = images.clone()
  fig = plt.figure(figsize=(224, 224))

  fig.add_subplot(1, 6, 1)
  plt.imshow(np.array(image_[5].cpu()))

  fig.add_subplot(1, 6, 2)
  plt.imshow(np.array(image_[220].cpu()))

  fig.add_subplot(1, 6, 3)
  plt.imshow(np.array(image_[410].cpu()))

  fig.add_subplot(1, 6, 4)
  plt.imshow(np.array(image_[500].cpu()))

  fig.add_subplot(1, 6, 5)
  plt.imshow(np.array(image_[720].cpu()))

  fig.add_subplot(1, 6, 6)
  plt.imshow(np.array(image_[900].cpu()))

  print("Classes for above Images:", labels[5].cpu().item(),"\t", labels[220].cpu().item(),"\t", labels[410].cpu().item(),"\t\t" ,
        labels[500].cpu().item(),"\t\t", labels[720].cpu().item(),"\t\t", labels[900].cpu().item())

  plt.show()

def calculate_accuracy(images, y_pred, labels, size):
  
  y_pred = torch.argmax(y_pred, axis = 1)

  train_acc = ((torch.sum(y_pred == labels))/(size))*100
  return train_acc


categories = ['cat', 'crocodile', 'dog', 'elephant', 'giraffe', 'lion']

def confusion_matrix_display(y_pred, labels):
  

  cf_matrix = confusion_matrix(labels.cpu(), y_pred.cpu())
  df_cm = pd.DataFrame(cf_matrix/15 , index = [i for i in categories],
                      columns = [i for i in categories])

  return df_cm

#Function to Train the Model
def run_model(model, input_val, target, loss_fn, optimiser):
  prediction = model(input_val)
  loss = loss_fn(prediction, target)

  # backpropagate error and update weights
  optimiser.zero_grad()
  loss.backward()
  optimiser.step()
  return loss


def fine_tuning(model, num_features, device):
  # num_features = model.fc.in_features    #512
  model.fc = torch.nn.Linear(num_features,6)    #Total 6 class
  model.to(device=device);

  loss_type = torch.nn.CrossEntropyLoss()
  #Adjust Learning rate from here
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)     
  #Every 10 epochs our learning rate will be multiplied by gamma
  step_lr = lr_scheduler.StepLR(optimizer, step_size= 10, gamma = 0.5)   

  return model, loss_type, optimizer, step_lr

def saliency_maps(X, y, model):
    
    model.eval()
    X.requires_grad_()
    
    scores = model(X)
    scores = (scores.gather(1, y.view(-1, 1)).squeeze())
    scores.backward(torch.FloatTensor([1.0]*scores.shape[0]).to(device=X.device))
    
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)

    N = 6
    saliency = saliency.cpu().numpy()
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(np.array(X[i].permute(1, 2, 0).cpu().detach().numpy()).astype('uint8'))
        
        plt.axis('off')
        plt.title(categories[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()
    

class Cnn(nn.Module):
    def __init__(self, n_classes):
        super(Cnn, self).__init__()
                
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        
    
        self.conv_2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_1_1 = nn.BatchNorm2d(64)
        self.conv_2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_1_2 = nn.BatchNorm2d(64)
        
        self.conv_2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_2_1 = nn.BatchNorm2d(64)
        self.conv_2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_2_2_2 = nn.BatchNorm2d(64)
        
        self.conv_3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_3_1_1 = nn.BatchNorm2d(128)
        self.conv_3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3_1_2 = nn.BatchNorm2d(128)
        self.size_adj_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)
       
        self.conv_3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3_2_1 = nn.BatchNorm2d(128)
        self.conv_3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn_3_2_2 = nn.BatchNorm2d(128)
        
        self.conv_4_1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn_4_1_1 = nn.BatchNorm2d(256)
        self.conv_4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_4_1_2 = nn.BatchNorm2d(256)
        self.size_adj_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)
        
        self.conv_4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_4_2_1 = nn.BatchNorm2d(256)
        self.conv_4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn_4_2_2 = nn.BatchNorm2d(256)
        
        self.conv_5_1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn_5_1_1 = nn.BatchNorm2d(512)
        self.conv_5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_5_1_2 = nn.BatchNorm2d(512)
        self.size_adj_5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2)
        
        self.conv_5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_5_2_1 = nn.BatchNorm2d(512)
        self.conv_5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_5_2_2 = nn.BatchNorm2d(512)
         
        self.ap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=1000)
        self.out = nn.Linear(in_features=1000, out_features=n_classes)

        self.relu = nn.ReLU()
        
    
    def forward(self, x):


        output_1 = self.mp_1(self.relu(self.bn_1(self.conv_1(x))))
            
        x = self.bn_2_1_2(self.conv_2_1_2(self.relu(self.bn_2_1_1(self.conv_2_1_1(output_1))))) 
        
        output_2_1 = self.relu(x + output_1)
         
        x = self.bn_2_2_2(self.conv_2_2_2(self.relu(self.bn_2_2_1(self.conv_2_2_1(output_2_1)))))   
        
        output_2 = self.relu(x + output_2_1)

        x = self.bn_3_1_2(self.conv_3_1_2(self.relu(self.bn_3_1_1(self.conv_3_1_1(output_2)))))   
        
        output_2 = self.size_adj_3(output_2) 
        
        output_3_1 = self.relu(x + output_2)
        
        x = self.bn_3_2_2(self.conv_3_2_2(self.relu(self.bn_3_2_1(self.conv_3_2_1(output_3_1)))))      
        
        output_3 = self.relu(x + output_3_1)
            
        x = self.bn_4_1_2(self.conv_4_1_2(self.relu(self.bn_4_1_1(self.conv_4_1_1(output_3)))))                 
        
        output_3 = self.size_adj_4(output_3) 
        
        output_4_1 = self.relu(x + output_3)
         
        x = self.bn_4_2_2(self.conv_4_2_2(self.relu(self.bn_4_2_1(self.conv_4_2_1(output_4_1)))))                 
        
        output_4 = self.relu(x + output_4_1)
 
        x = self.bn_5_1_2(self.conv_5_1_2(self.relu(self.bn_5_1_1(self.conv_5_1_1(output_4)))))                 
        
        output_4 = self.size_adj_5(output_4)
        
        output_5_1 = self.relu(x + output_4)

        x = self.bn_5_2_1(self.conv_5_2_1(self.relu(self.bn_5_2_1(self.conv_5_2_1(output_5_1)))))                
        
        output_5 = self.relu(x + output_5_1)

        x = self.ap(output_5)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc(x))
        x = self.out(x)

        return x


