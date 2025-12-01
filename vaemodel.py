import torch 

class VAE_Encoder(torch.nn.Module) :
    def __init__ (self, ch_base, dim, im_size, device,  **kwargs) :        
        super().__init__()
        self.convs = [] 
        self.lns = []        
        in_channels = 1
        out_channels = ch_base

        self.convs.append(torch.nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, padding = 'same', device = device))                    
        self.lns.append(torch.nn.BatchNorm2d(out_channels, device = device))
        #self.lns.append(torch.nn.LayerNorm(normalized_shape = (out_channels, im_size, im_size), device = device))
        
        for i  in range(1, 3) :            
            in_channels = out_channels
            out_channels=ch_base*(2**i)                         
            self.convs.append(torch.nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, padding = 'same', device = device))            
            self.lns.append(torch.nn.BatchNorm2d(out_channels, device = device))            
            #self.lns.append(torch.nn.LayerNorm(normalized_shape = (out_channels, (im_size - 1)//(2**i), (im_size - 1)//(2**i)), device = device))            

            
        self.projection_mu = torch.nn.Linear(out_channels, dim, device = device)
        self.projection_sigma = torch.nn.Linear(out_channels, dim, device = device)
        
    def forward(self, input) :
        x = input
        for i, conv in enumerate(self.convs) :
            x = conv(x)
            x = self.lns[i](x)
            x = torch.nn.ReLU()(x)
            x = torch.nn.MaxPool2d((3,3), stride = 2)(x)
        #global average pooling    
        x = x.mean(dim = (2,3))
        mu = self.projection_mu(x)
        log_sigma = self.projection_sigma(x)
        return mu, log_sigma
    

class VAE_Decoder(torch.nn.Module) :
    def __init__(self, ch_base, dim, im_size, device) :        
        super().__init__()     
        self.imsize_0 = im_size // (2**2)
        self.proyection = torch.nn.Linear(dim, self.imsize_0*self.imsize_0)
        self.tconvs = [] 
        self.lns = []        
        
        ch_base = 32
        in_channels = 1

        dim_d = self.imsize_0
        out_channels = 0
        self.dims = []
        out_channels = 32
        for i  in range(0,2) :
            dim_d = dim_d * 2
            #out_channels=ch_base*(2**i)
            
            self.tconvs.append(torch.nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, stride = 2, output_padding= 1, padding = 1, device = device))
            self.lns.append(torch.nn.BatchNorm2d(out_channels, device = device))
            #self.lns.append(torch.nn.LayerNorm(normalized_shape = (out_channels, dim_d,dim_d), device = device))
            self.dims.append((dim_d,dim_d))
            in_channels = out_channels

        self.conv = torch.nn.Conv2d(in_channels = out_channels, out_channels = 1, kernel_size=1, device = device)
        
    def forward(self, input) :
        x = input
        x = self.proyection(x)
        x = x.view(-1, 1, self.imsize_0, self.imsize_0)
        
        for i, tconv in enumerate(self.tconvs) :
            x = tconv(x)
            x = self.lns[i](x)
            x = torch.nn.ReLU()(x)
        y = self.conv(x)
        y = torch.nn.Sigmoid()(y)
        return y
        
class VAE(torch.nn.Module) :
    def __init__(self, ch_base, dim, im_size, device) :
        super().__init__()     
        self.device = device
        self.encoder = VAE_Encoder(ch_base, dim, im_size, device)
        self.decoder = VAE_Decoder(ch_base, dim, im_size,  device)
        self.dim = dim     

    def sampling(self, mu, log_sigma) :            
        epsilon = torch.randn(mu.shape).to(self.device)
        return mu + torch.exp(log_sigma) * epsilon
    
    def forward(self, input) :
        mu, log_sigma = self.encoder(input)
        z = self.sampling(mu, log_sigma)
        x = self.decoder(z)
        return mu, log_sigma, x
    
    def kl_loss(self, mu, log_sigma) :
        loss =0.5*(-torch.sum(log_sigma, dim = 1) + torch.sum(torch.exp(log_sigma),dim = 1) + torch.sum(torch.square(mu), dim = 1))        
        return loss
    
    def bce_loss(self ,inputs, outputs) :
        return - (inputs * torch.log(outputs + 10e-10) + (1 - inputs) * torch.log(1 - outputs + 10e-10))

    def fit(self, tr_dataset, val_dataset, optimizer, epochs = 100, device = 'cuda') :
        STEPS_STATS = int(0.1*len(tr_dataset))
        #each epoch
        reconstruction_loss = torch.nn.BCELoss()        
        def train_one_epoch(epoch) :
            running_loss = 0.        
            last_loss = 0.
            for i, data in enumerate(tr_dataset):
                # every data instance is an input + label pair
                inputs = data                                
                if next(self.parameters()).is_cuda :
                    inputs  = inputs.to(device)                    
                # set your gradients to zero for every batch
                optimizer.zero_grad()
                # forward phase -> making predictions for this batch 
                mu, sigma, outputs = self.predict(inputs)
                # compute the loss and its gradients
                outputs = torch.squeeze(outputs, dim = 1)
                #r_loss = reconstruction_loss(inputs, outputs)
                r_loss = self.bce_loss(inputs, outputs)
                kl_l = self.kl_loss(mu, sigma) 
                loss = torch.mean(r_loss) + torch.mean(kl_l)
                #compute gradientes using backpropagation
                loss.backward()
                # adjust learning weights
                optimizer.step()
                # gather stats and report
                running_loss += loss.item()                                
                if i % STEPS_STATS == (STEPS_STATS - 1) :
                    last_loss = running_loss / STEPS_STATS # loss per batch                    
                    print('  batch {} loss: {} '.format(i + 1, last_loss))        
                    running_loss = 0.                    
            return last_loss
                
        best_vloss = 1_000_000.
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch + 1))
            # make sure gradient tracking is on, and do a pass over the data
            self.train()
            avg_loss = train_one_epoch(epoch + 1)
            running_vloss = 0.0            
            # set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.eval()

            # disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(val_dataset):
                    vinputs = vdata                    
                    if next(self.parameters()).is_cuda :
                        vinputs  = vinputs.to(device)                        
                        
                    vmu, vlog_sigma, voutputs = self.predict(vinputs)                     
                    voutputs = torch.squeeze(voutputs, dim = 1)
                    #r_loss = reconstruction_loss(vinputs, voutputs)
                    r_loss = self.bce_loss(vinputs, voutputs)
                    kl_l = self.kl_loss(vmu, vlog_sigma) 
                    vloss = torch.mean(r_loss) + torch.mean(kl_l)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            #avg_vacc = running_vacc / (i + 1)
            print(' TRAIN: [loss {}] VAL : [loss {}]'.format(avg_loss, avg_vloss))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'vae_mnist_model'
                torch.save(self.state_dict(), model_path)    

    #eval
    def predict(self, inputs) :
        inputs = torch.Tensor.unsqueeze(inputs, dim = 1)
        pred = self(inputs)
        return pred
    
    def generate(self) :
        z = torch.randn(self.dim).to(self.device)
        y = self.decoder(z)
        return y
