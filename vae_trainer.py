import torch
import torchdiff
import torchdiff.ldm as ldm
from torchdiff.utils import Metrics
from torchdiff.ldm import AutoencoderLDM, TrainAE

#dataloader
import os 
print(os.path.join(os.getcwd(), 'mnist'))
import torch.utils.data as data
import dataloader


#download dataset
#clone machine learning repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#1: dataset loader (defining dataset)
#datapath = '/hd_data/MNIST-5000'
#datapath = '/hd_data/mnist'
datapath = '/hd_data/MNIST-5000'
tr_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'train')
tr_dataset = data.DataLoader(tr_dataset, batch_size = 8, shuffle = True)

val_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'valid') 
val_dataset = data.DataLoader(val_dataset, batch_size = 8, shuffle = False)

# for d in val_dataset :
#     print(d[1].shape)
# Initialize VAE with optimized hyperparameters
vae = AutoencoderLDM(
    in_channels=3,                      # RGB input
    down_channels=[16, 32, 64],  # Encoder channel progression
    up_channels=[64, 32, 16],    # Decoder channel progression  
    out_channels=3,                     # RGB output
    dropout_rate=0.2,                   # Regularization
    num_heads=4,                        # Multi-head attention
    num_groups=1,                      # Group normalization
    num_layers_per_block=2,            
    total_down_sampling_factor=8,       
    latent_channels=2,                  # Latent channel dimension
    num_embeddings=32,                # VQ codebook size
    use_vq=True,                        
    beta=1e-6                           # KL weight
)

# Configure optimizer with weight decay
optimizer = torch.optim.AdamW(
    vae.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

# Initialize comprehensive metrics
metrics = Metrics(
    device="cuda",
    fid=False,           # Fréchet Inception Distance
    #ssim=True,          # Structural Similarity Index
    lpips_=True        # Perceptual similarity
   # psnr=True          # Peak Signal-to-Noise Ratio
)

# Setup training configuration
vae_trainer = ldm.TrainAE(
    model=vae,
    optimizer=optimizer,
    data_loader=tr_dataset,
    val_loader=val_dataset,
    max_epochs=500,
    metrics_=metrics,
    device="cpu",
    store_path="models/vae_ldm.pth",
    val_frequency=10,
    checkpoint=50
    #gradient_clip_norm=1.0
)

# Execute training
#vae_trainer()
ck = torch.load('models/vae_ldm.pth/ldm_epoch_10.pth', map_location = 'cpu', weights_only=True)
state_dict = ck['model_state_dict']
vae.load_state_dict(state_dict)

        

#vae_trainer.load_checkpoint('models/vae_ldm.pth/ldm_epoch_10.pth')
data_iterator = iter(val_dataset)
# Get the next batch of data
batch_features, batch_labels = next(data_iterator)
for batch in val_dataset :
    for item in batch[0] :
        #print(torch.unsqueeze(item, dim = 0))
        z, loss = vae.encode(torch.unsqueeze(item, dim = 0))
        y = vae.decode(z)
        print(y)

