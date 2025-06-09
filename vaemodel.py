import torch 


class VAE_Encoder(torch.nn.Module) :
    def __init__ (self, ch_base, dim, im_size, **kwargs) :        
        super().__init__()     
        self.convs = [] 
        self.lns = []        
        in_channels = 1
        out_channels = 0
        for i  in range(3) :            
            out_channels=ch_base*(2**i)
            self.convs.append(torch.nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, padding = 'same'))
            self.lns.append(torch.nn.LayerNorm(normalized_shape = (im_size//(2**i),im_size//2**i)))
            in_channels = out_channels 
            
        self.projection_mu = torch.nn.Linear(out_channels, dim)
        self.projection_sigma = torch.nn.Linear(out_channels, dim)
        
    def forward(self, input) :
        x = input
        for i, conv in enumerate(self.convs) :
            x = conv(input)
            x = self.lns[i](x)
            x = torch.nn.ReLU()(x)
        x = x.mean(dim = (2,3))
        mu = self.projection_mu(x)
        log_sigma = self.projection_sigma(x)
        return mu, log_sigma
    

class VAE_Decoder(torch.nn.Module) :
    def __init__(self, ch_base, dim, im_size) :        
        super().__init__()     
        self.proyection = torch.nn.Linear(dim, dim*dim)
        self.tconvs = [] 
        self.lns = []        
        ch_base = 32
        in_channels = 1

        dim_d = dim 
        out_channels = 0
        for i  in range(3) :
            out_channels=ch_base*(2**i)
            self.tconvs.append(torch.nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, padding = 'same'))
            self.lns.append(torch.nn.LayerNorm(normalized_shape = (dim_d,dim_d)))
            in_channels = out_channels
            dim_d = dim_d*2

        self.conv = torch.nn.Conv2d(in_channels = out_channels, out_channels = 1, kernel_size=1, padding = 'same')
        
    def forward(self, input) :
        x = input
        x = self.proyection(x)
        x = x.view(-1, 1, self.dim, self.dim)
        
        for i, tconv in enumerate(self.tconvs) :
            x = tconv(x)
            x = self.lns[i](x)
            x = torch.nn.ReLU()(x)
        y = self.conv(x)
        y = torch.nn.Sigmoid(x)
        return y
        
class VAE(torch.nn.Module) :
    def __init__(self, ch_base, dim, im_size) :
        super().__init__()     
        self.encoder = VAE_Encoder(ch_base, dim, im_size)
        self.decoder = VAE_Decoder(ch_base, dim, im_size)
        
    def sampling(self, mu, var_log) :            
        epsilon = torch.randn(mu.shape)
        return mu + torch.exp(var_log) * epsilon
    
    def forward(self, input) :
        mu, log_sigma = self.encoder(input)
        z = self.sampling(mu, log_sigma)
        x = self.decoder(z)
        return mu, log_sigma, x
    
    def lk_loss(self, mu, log_sigma) :
        loss =-0.5 * (-torch.sum(log_sigma, dim = 1) + torch.prod(log_sigma,dim = 1) + torch.sum(torch.square(mu), dim = 1))
        
        return loss
        

    def fit(self, tr_dataset, val_dataset, optimizer, epochs = 100, device = 'cuda') :
        STEPS_STATS = int(0.1*len(tr_dataset))
        #each epoch
        reconstruction_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss
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
                r_loss = reconstruction_loss(outputs, inputs)
                kl_loss = kl_loss(mu, sigma) 
                loss = torch.mean(r_loss + kl_loss)                
                #compute gradientes using backpropagation
                loss.backward()
                # adjust learning weights
                optimizer.step()
                # gather stats and report
                running_loss += loss.item()
                acc = torch.mean(torch.eq(torch.argmax(outputs, dim = 1), tr_labels).float())
                running_acc += acc
                if i % STEPS_STATS == (STEPS_STATS - 1) :
                    last_loss = running_loss / STEPS_STATS # loss per batch
                    last_acc  = running_acc / STEPS_STATS # acc per batch
                    print('  batch {} loss: {} acc: {}'.format(i + 1, last_loss, last_acc))        
                    running_loss = 0.
                    running_acc = 0.            
            return last_loss, last_acc 
                
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
                    r_loss = reconstruction_loss(outputs, inputs)
                    kl_loss = kl_loss(mu, sigma) 
                    loss = torch.mean(r_loss + kl_loss)                                    
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            avg_vacc = running_vacc / (i + 1)
            print(' TRAIN: [loss {} acc {}] VAL : [loss {} acc {}]'.format(avg_loss, avg_acc, avg_vloss, avg_vacc))

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