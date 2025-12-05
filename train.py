import os 
print(os.path.join(os.getcwd(), 'mnist'))
import torch.utils.data as data
import dataloader
import torch 
import vaemodel

#download dataset
#clone machine learning repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#1: dataset loader (defining dataset)
datapath = '/hd_data/MNIST-5000'
#datapath = '/hd_data/mnist'
#datapath = '/home/jmsaavedrar/Documents/mnist'

tr_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'train')
tr_dataset = data.DataLoader(tr_dataset, batch_size = 64, shuffle = True)

val_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'valid')
val_dataset = data.DataLoader(val_dataset, batch_size = 64, shuffle = False)

#2: defining the model
model = vaemodel.VAE(ch_base = 16, dim = 32, im_size = 32, device = device)
if device == torch.device('cuda') :    
    model.to(device)    

# defining loss and optimize
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters())

model.fit(tr_dataset, val_dataset, optimizer, device = device, epochs = 500)
