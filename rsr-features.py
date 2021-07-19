# -*- coding: utf-8 -*-
"""
RSR Algorithm by Jose Lezama <jlezama@fing.edu.uy>

Adapted from DCGAN and E2GAN
"""

import sys, os

import imageio

import cfg
import models_search
# from functions import validate
from utils.utils import set_log_dir, create_logger

# if not os.path.isfile('gan-vae-pretrained-pytorch'):
#   os.system('git clone https://github.com/csinva/gan-vae-pretrained-pytorch.git')

# os.chdir('/home/jose/code/20201001_dcgan_rsr/gan-vae-pretrained-pytorch/cifar10_dcgan')


# if not os.path.isfile('pytorch-fid'):
#   os.system('git clone https://github.com/mseitzer/pytorch-fid.git')

sys.path.append('pytorch-fid/pytorch_fid')

from inception import InceptionV3

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
inception_model = InceptionV3([block_idx], normalize_input=False)
# print('Number of model parameters: {}'.format(
#     sum([p.data.nelement() for p in inception_model.parameters()])))
# exit()

inception_model.cuda()

inception_model.eval()

"""Download statistics and compute whitening matrix (cholesky of inverse) so that 
$$W.(x-\mu) \sim \mathcal{N}(0,I)$$
"""

def preprocess_for_inception(img):
   #return (img+1.0)/2.0
   return img
import numpy as np
if not os.path.isfile('fid_stats_cifar10_train.npz'):
   os.system('wget http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz')
precomputed_fid_stats = np.load('fid_stats_cifar10_train.npz')

# option 1 cholesky of inverse
# sigma_inverse = np.linalg.inv(precomputed_fid_stats['sigma'])
# W = np.linalg.cholesky(sigma_inverse) # whitening matrix

# option 2 inverse of cholesky
Q = np.linalg.cholesky(precomputed_fid_stats['sigma'])
W = np.linalg.inv(Q)

# Commented out IPython magic to ensure Python compatibility.
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import pylab
import numpy as np
# %load_ext autoreload
# %autoreload 2

num_gpu = 1 if torch.cuda.is_available() else 0

# load the E2GAN models
args = cfg.parse_args()

args.img_size = 32
args.bottom_width = 4
args.gen_model = 'shared_gan_leaky'
args.latent_dim = 128
args.gf_dim = 256 
args.g_spectral_norm = False 
args.load_path = 'checkpoints/e2gan_cifar.pth'

args.arch = [0, 1, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 1, 2] # e2gan architecture is defined this way (see paper)

G = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
G.set_arch(args.arch, cur_stage=2)

# print('Number of model parameters: {}'.format(
#     sum([p.data.nelement() for p in G.parameters()])))
# exit()

# load weights
checkpoint_file = args.load_path
assert os.path.exists(checkpoint_file)
checkpoint = torch.load(checkpoint_file)

if 'avg_gen_state_dict' in checkpoint:
   G.load_state_dict(checkpoint['avg_gen_state_dict'])
   epoch = checkpoint['epoch'] - 1
   print(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
else:
   G.load_state_dict(checkpoint)
   print(f'=> loaded checkpoint {checkpoint_file}')
                               
print(G)

G.train()


description='full_lbs_%i_%i' % (args.lbs_init, args.lbs_end)

print('running experiment %s' % description)

outdir_images = 'outimgs/fakes/%s/' % (description)
outdir_weights = 'weights/%s/' % (description)

os.system('mkdir -p %s' % outdir_images)
os.system('mkdir -p %s' % outdir_weights)

   
if torch.cuda.is_available():
    G = G.cuda()



batch_size =  args.gen_batch_size
latent_size = args.latent_dim

small_batch_size = batch_size
large_batch_size = args.lbs_init

large_batch_size_init = large_batch_size
large_batch_size_end =  args.lbs_end + batch_size

max_epoch = args.max_epoch

Nsteps = (large_batch_size_end-large_batch_size_init)//small_batch_size
step_length = max_epoch//Nsteps

NS = large_batch_size//small_batch_size


N_rotmat = 4000 # number of random projections
d_img = 2048 # inception feature dimension


G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, weight_decay=0) # set LR and weight decay here

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_opt, max_epoch)

if args.dataset == "LSUN_bedrooms":
   dataset = dset.LSUN(root='.data/',
                       classes=['bedroom_train'],
                        transform=transforms.Compose([
                           # transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == "LSUN_churches":
   dataset = dset.LSUN(root='.data/',
                          classes=['church_outdoor_train'],
                        transform=transforms.Compose([
                           transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == "celeba":
   dataset = dset.CelebA(root='.data/', download=True,
                        transform=transforms.Compose([
                           # transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == "imagenet":
   dataset = dset.ImageNet(root='.data/', download=True,
                        transform=transforms.Compose([
                           # transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
else:
   dataset = dset.CIFAR10(root='.data/', download=True,
                        transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
nc=3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=small_batch_size,
                                         shuffle=True, num_workers=2, drop_last=True)

dataloader_iterator = iter(dataloader)


losses = []

try_gt = False # set to True to use real samples as fake samples, helps to get an upper bound on performance

for epoch in range(max_epoch): 

  scheduler.step() 
   
  new_large_batch_size = large_batch_size_init + (epoch//step_length)*small_batch_size

  # enarge large_batch_size accoring to lbs_init and lbs_end
  if  new_large_batch_size != large_batch_size:
     large_batch_size = new_large_batch_size
     threshold, std_threshold = compute_threshold(large_batch_size)
     threshold += args.thresh_std * std_threshold

     NS = large_batch_size//small_batch_size
     print('epoch %i, new large batch size is %i, threshold is %f, NS is %i' % (epoch, new_large_batch_size, threshold, NS))
     new_lbs = 1
  else:
     new_lbs = 0
     
  if epoch==0: # first iteration uses random projections
     rotmat_img = torch.randn(d_img, N_rotmat).cuda()
     rotmat_img = rotmat_img/torch.sqrt(torch.sum(rotmat_img**2, dim=0))

  elif epoch >0: # following iterations uses pairs and worst projections from previous iteration
   with torch.no_grad():
     pworst = 1/3.0
     # keep 1/3rd of worst projections, add 2/3rd new ones
     
     worst_values_img, worst_index_img = torch.sort(G_loss_all_img, descending=True)
     rotmat_img_prev = rotmat_img[:,worst_index_img[:int(N_rotmat*pworst)]]

     # rotmatimg will be taken from pairs of gt, output
     N_rotmat_new = int(N_rotmat*(1-pworst))
     ix_gt = np.random.randint(0,large_batch_size - small_batch_size*new_lbs, N_rotmat_new)
     ix_output = np.random.randint(0,large_batch_size - small_batch_size*new_lbs, N_rotmat_new)

     vectors_gt = all_gt[ix_gt, :].detach().t().cuda()
     vectors_out = all_output_img[ix_output, :].detach().t()

     rotmat_img = (vectors_gt-vectors_out)

     # worst_image_np = rotmat_img[:,0].cpu().detach().numpy()
     # worst_image_np = worst_image_np.reshape(3, 32, 32)
     # worst_image_np = ((worst_image_np.transpose((1, 2, 0))/2.0 + .5)*255).astype(np.uint8)
     # imageio.imwrite('%s/worst_img_%06i.png' % (outdir_images, epoch), worst_image_np)
     
     # normalize
     rotmat_img = rotmat_img/torch.sqrt(torch.sum(rotmat_img**2, dim=0))
     rotmat_img = torch.cat((rotmat_img, rotmat_img_prev), dim=1)     

     print('DEBUG: worst values img', worst_values_img[:15])
                                     

  # initialize tensors for noise vectors, real data and fake data
  
  all_z = torch.randn(large_batch_size, latent_size).cuda()
  all_gt = torch.zeros(large_batch_size, d_img).cuda()
  all_output_img = torch.zeros(large_batch_size, d_img).cuda()

  gt_images = [] # auxiliary list for use_gt = True


  ####################################################
  # STEP 1. RUN 
  
  with torch.no_grad():
    for idx in range(NS):

      if epoch>=0: 
          try:
             images, _ = next(dataloader_iterator)
          except:
             dataloader_iterator = iter(dataloader)
             images, _ = next(dataloader_iterator)
          images = images.cuda()
          inception_features_gt = inception_model(images)[0].view(batch_size, -1)
          all_gt[idx*batch_size:(idx+1)*batch_size,:] = inception_features_gt
    
      z = all_z[idx*batch_size:(idx+1)*batch_size,:]

      if try_gt: # for debugging
         try:
            images, _ = next(dataloader_iterator)
         except:
            dataloader_iterator = iter(dataloader)
            images, _ = next(dataloader_iterator)
         fake_images = images.cuda()
         gt_images.append(fake_images)
         
      else:
         fake_images = G(z)
      
      # compute inception feature
      inception_features = inception_model(preprocess_for_inception(fake_images))[0].view(batch_size, -1)

      # all_output_img[idx*batch_size:(idx+1)*batch_size,:] = fake_images.view(batch_size,-1) # for raw pixel values use this
      all_output_img[idx*batch_size:(idx+1)*batch_size,:] = inception_features


      
  ## finished computing features, now project
  with torch.no_grad():
     all_output_img_projected = all_output_img.mm(rotmat_img)
     all_gt_projected = all_gt.mm(rotmat_img)


  ####################################################
  # STEP 2. SORT
  

  with torch.no_grad():
     # move to cpu, sort, move back to gpu
     # [_, out_img_sort_ix] = torch.sort(all_output_img_projected.cpu(), dim=0)
     # out_img_sort_relative = out_img_sort_ix.argsort(0)
     # out_img_sort_relative = out_img_sort_relative.cuda()
     [_, out_img_sort_ix] = torch.sort(all_output_img_projected, dim=0)
     out_img_sort_relative = out_img_sort_ix.argsort(0)

     # [gt_sort_val, _] = torch.sort(all_gt_projected.cpu(), dim=0)
     # gt_sort_val = gt_sort_val.cuda()
     [gt_sort_val, _] = torch.sort(all_gt_projected, dim=0)

  
  ####################################################
  # STEP 3. RE-RUN


  # initialize gradient                                                                      
  G_opt.zero_grad()
  full_batch_loss = 0

  G_loss_all_feat = 0
  G_loss_all_img = 0

  SQRT2 = 1.4142135623731

  # now do actual comparison 
  for idx in range(0,large_batch_size,small_batch_size):
      z = all_z[idx:idx+small_batch_size,:]

      if try_gt:
         print('try gt!')
         fake_images = gt_images[idx//small_batch_size] 
      else:
         fake_images = G(z)

      # compute inception feature
      inception_features = inception_model(preprocess_for_inception(fake_images))[0].view(batch_size, -1)

      output_img = inception_features.mm(rotmat_img) # project


      # get the relative position of the output
      rel_ix_img = out_img_sort_relative[idx:idx+small_batch_size,:]

      # now get the equivalent positions of the gt
      gt = gt_sort_val.gather(0, rel_ix_img).cuda()

      diff_img = (gt-output_img)**2


      threshold_img = 1e-4 # don't penalize too small differences, this is normal even for samples of the same distribution. Trying other values for this hyperparameter might be interesing
      diff_img = (torch.clamp(diff_img, min=threshold_img)-threshold_img)


      G_loss_row_img = torch.sum(diff_img, dim=0) / large_batch_size 
      G_loss_img = torch.sum(G_loss_row_img)/ rotmat_img.shape[1] 
         

      # print('DEBUG: dist loss img: %f' % (G_loss_img.item()))
      # print('DEBUG: --')

      G_loss =  G_loss_img

      if not try_gt:
         G_loss.backward()

      G_loss_all_img += G_loss_row_img.detach().cpu()


      full_batch_loss += G_loss.item()

  if not try_gt: #epoch>1: 
     G_opt.step()

  losses.append(full_batch_loss)

  ## RSR ENDS HERE

  # what follows is for logging/saving/debugging
  
  if 1:
    print('DEBUG: large_batch_size', large_batch_size, 'epoch', epoch, 'loss', losses[-1],  'lr', scheduler.get_lr())


  if epoch % 50 ==0 and epoch >0:
  #if epoch % 1 ==0 and epoch >0:
    if epoch % 100  == 0:
       # save model
       torch.save(G.state_dict(), '%s/G_%06i.pth' % (outdir_weights, epoch))

    count_imgs = 0
    G.eval()
    with torch.no_grad():
     Nb = 11000//batch_size
     for i in range(Nb):
        print('processing batch %i of %i' % (i, Nb))
        z = torch.randn(batch_size, latent_size).cuda()
        fake_images = G(z)
  
        fake_images_np = fake_images.cpu().detach().numpy()
  
        fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, 32, 32)
        fake_images_np = ((fake_images_np.transpose((0, 2, 3, 1))/2.0 + .5)*255).astype(np.uint8)
  
        for i in range(batch_size):
           imageio.imwrite('%s/img_%06i.png' % (outdir_images, count_imgs), fake_images_np[i])
           count_imgs+=1

    
    G.train()
    print('wrote images to %s' % outdir_images)

    torch.cuda.empty_cache()

    ###################
    # Compute FID score
    # requires https://github.com/mseitzer/pytorch-fid
    fid_command = 'python ../pytorch-fid/pytorch_fid/fid_score.py %s fid_stats_cifar10_train.npz  --device cuda:0 ' % outdir_images
    os.system(fid_command)

    
    # END RSR

