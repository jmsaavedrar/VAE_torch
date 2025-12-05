import torch
import torchdiff.ldm as ldm
from torchdiff.utils import Metrics
from torchdiff.ldm import AutoencoderLDM, TrainAE

#dataloader
import os 
print(os.path.join(os.getcwd(), 'mnist'))
import torch.utils.data as data
import dataloader
import numpy as np
import matplotlib.pyplot as plt

#download dataset
#clone machine learning repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#1: dataset loader (defining dataset)
datapath = '/hd_data/MNIST-5000'
#datapath = '/hd_data/mnist'
#datapath = '/home/jmsaavedrar/Documents/mnist'
tr_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'train')
tr_dataset = data.DataLoader(tr_dataset, batch_size = 32, shuffle = True)

val_dataset = dataloader.MNIST_Dataloader(datapath, datatype = 'valid') 
val_dataset = data.DataLoader(val_dataset, batch_size = 32, shuffle = False)

# for d in val_dataset :
#     print(d[1].shape)
# Initialize VAE with optimized hyperparameters
vae = AutoencoderLDM(
    in_channels=1,                      # RGB input
    down_channels=[16, 32, 64],         # Encoder channel progression
    up_channels=[64, 32, 16],           # Decoder channel progression  
    out_channels=1,                     # RGB output
    dropout_rate=0.1,                   # Regularization
    num_heads=2,                        # Multi-head attention
    num_groups=1,                      # Group normalization
    num_layers_per_block=2,
    total_down_sampling_factor=8,
    latent_channels=2,                  # Latent channel dimension
    num_embeddings=128,                # VQ codebook size
    use_vq=False,                       #KL_divergence 
    beta=1e-2
)

# Configure optimizer with weight decay
# optimizer = torch.optim.AdamW(
#     vae.parameters(),
#     lr=2e-4,
#     betas=(0.9, 0.999),
#     weight_decay=1e-4
# )

optimizer = torch.optim.Adam(
    vae.parameters(),
    lr=2e-4
    # betas=(0.9, 0.999),
    # weight_decay=1e-4
)

# Initialize comprehensive metrics
metrics = Metrics(
    device="cuda",
    fid=False,           # Fréchet Inception Distance
    metrics=True,          # Structural Similarity Index
    lpips_=False        # Perceptual similarity
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
    device=device,
    store_path="models/vae_ldm.pth",
    val_frequency=10,
    checkpoint=50,
    grad_accumulation_steps = 1, 
    patience=50,
    kl_warmup_epochs = 10
    #gradient_clip_norm=1.0
)

# Execute training
mode = 'test'
if mode == 'train' :
    vae_trainer()
else :
    ck = torch.load('models/vae_ldm.pth/ldm_epoch_360.pth', map_location = device, weights_only=True)
    state_dict = ck['model_state_dict']
    vae.load_state_dict(state_dict)

    #filename = '/hd_data/MNIST-5000/valid_images/digit_mnist_00050_4.png'   
    #filename = '/hd_data/MNIST-5000/valid_images/digit_mnist_00131_6.png'
    filename = '/hd_data/MNIST-5000/train_images/digit_mnist_14712_5.png'
    image =dataloader.read_image(filename)
    image = np.expand_dims(image, axis = 0)
    #z, loss = vae.encode(torch.Tensor(image).to(device))    
    z = torch.normal(mean = 0.0, std=1.0, size = (1,2,8,8)).to(device)
    print(z[0,:,:1,:1])
    y = vae.decode(z)
    #y = torch.nn.Sigmoid()(y)
    fig, xa = plt.subplots(1,2)
    xa[0].imshow(np.transpose(dataloader.rever_trans(image[0,:,:,:]), (1,2,0)))
    xa[1].imshow(np.transpose(dataloader.rever_trans(y.cpu().detach().numpy()[0,:,:,:]), (1,2,0)))
    plt.show()
