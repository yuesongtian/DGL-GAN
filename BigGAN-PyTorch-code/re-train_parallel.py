""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from tensorboardX import SummaryWriter
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback

# Import DDP Module
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

arch_woDeconv_sep = {'cell_0': {'up_0': 'bilinear_conv', 'up_1': 'bilinear_conv', 'normal_0': 'conv_5x5', 'normal_1': 'sep_conv_7x7'}, 'cell_1': {'up_0': 'nearest_conv', 'up_1': 'bilinear_conv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}, 'cell_2': {'up_0': 'bilinear_conv', 'up_1': 'nearest_conv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}} 
arch_woDeconv_sep_WoLea = {'cell_0': {'up_0': 'bilinear', 'up_1': 'bilinear', 'normal_0': 'conv_5x5', 'normal_1': 'sep_conv_7x7'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}, 'cell_2': {'up_0': 'bilinear', 'up_1': 'nearest', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}}

arch_Deconv_WoSep = {'cell_0': {'up_0': 'deconv', 'up_1': 'deconv', 'normal_0': 'conv_1x1', 'normal_1': 'conv_3x3'}, 'cell_1': {'up_0': 'nearest_conv', 'up_1': 'deconv', 'normal_0': 'conv_1x1', 'normal_1': 'conv_1x1'}, 'cell_2': {'up_0': 'bilinear_conv', 'up_1': 'deconv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}}
arch_Deconv_WoSep_WoLea = {'cell_0': {'up_0': 'deconv', 'up_1': 'deconv', 'normal_0': 'conv_1x1', 'normal_1': 'conv_3x3'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'deconv', 'normal_0': 'conv_1x1', 'normal_1': 'conv_1x1'}, 'cell_2': {'up_0': 'bilinear', 'up_1': 'deconv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}}

arch_woDeconv_woSep = {'cell_0': {'up_0': 'nearest_conv', 'up_1': 'nearest_conv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_1x1'}, 'cell_1': {'up_0': 'nearest_conv', 'up_1': 'nearest_conv', 'normal_0': 'conv_1x1', 'normal_1': 'conv_3x3'}, 'cell_2': {'up_0': 'nearest_conv', 'up_1': 'bilinear_conv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}}
arch_woDeconv_woSep_WoLea = {'cell_0': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_5x5', 'normal_1': 'conv_1x1'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_1x1', 'normal_1': 'conv_3x3'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}}

arch_Deconv_Sep = {'cell_0': {'up_0': 'bilinear_conv', 'up_1': 'deconv', 'normal_0': 'conv_3x3', 'normal_1': 'conv_3x3'}, 'cell_1': {'up_0': 'deconv', 'up_1': 'deconv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_1x1'}, 'cell_2': {'up_0': 'bilinear_conv', 'up_1': 'bilinear_conv', 'normal_0': 'sep_conv_7x7', 'normal_1': 'conv_5x5'}}
arch_Deconv_Sep_WoLea = {'cell_0': {'up_0': 'bilinear', 'up_1': 'deconv', 'normal_0': 'conv_3x3', 'normal_1': 'conv_3x3'}, 'cell_1': {'up_0': 'deconv', 'up_1': 'deconv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_1x1'}, 'cell_2': {'up_0': 'bilinear', 'up_1': 'bilinear', 'normal_0': 'sep_conv_7x7', 'normal_1': 'conv_5x5'}}
arch_Deconv_Sep_WoLea_l = {'cell_0': {'up_0': 'bilinear', 'up_1': 'deconv', 'normal_0': 'conv_3x3', 'normal_1': 'conv_3x3'}, 'cell_1': {'up_0': 'deconv', 'up_1': 'deconv', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}, 'cell_2': {'up_0': 'bilinear', 'up_1': 'bilinear', 'normal_0': 'sep_conv_7x7', 'normal_1': 'conv_5x5'}}

arch_Deconv_Sep_SearchWoLea = {'cell_0': {'up_0': 'bilinear', 'up_1': 'nearest', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_3x3'}, 'cell_1': {'up_0': 'bilinear', 'up_1': 'deconv', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_5x5'}, 'cell_2': {'up_0': 'deconv', 'up_1': 'deconv', 'normal_0': 'sep_conv_7x7', 'normal_1': 'conv_3x3'}}

arch_pretrainBar_woDeconvSep = {'cell_0': {'up_0': 'bilinear', 'up_1': 'bilinear', 'normal_0': 'sep_conv_5x5', 'normal_1': 'sep_conv_5x5'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_3x3'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'sep_conv_3x3', 'normal_1': 'sep_conv_3x3'}}

arch_WoDeconv_Sep_searchWoLea_former = {'cell_0': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_3x3', 'normal_1': 'conv_1x1'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_3x3', 'normal_1': 'conv_3x3'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'sep_conv_5x5', 'normal_1': 'conv_5x5'}}

arch_WoDeconv_Sep_searchWoLea_latter = {'cell_0': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_5x5', 'normal_1': 'conv_5x5'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_3x3', 'normal_1': 'sep_conv_5x5'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_1x1', 'normal_1': 'conv_1x1'}}

arch_woDeconv_Img = {'cell_0': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_7x7'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_1x1', 'normal_1': 'conv_1x1'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_3x3'}, 'cell_3': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_5x5', 'normal_1': 'conv_3x3'}, 'cell_4': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_3x3', 'normal_1': 'conv_5x5'}}

arch_woDeconv_Img_94000 = {'cell_0': {'up_0': 'bilinear', 'up_1': 'bilinear', 'normal_0': 'sep_conv_3x3', 'normal_1': 'sep_conv_7x7'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'sep_conv_7x7', 'normal_1': 'sep_conv_7x7'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_1x1', 'normal_1': 'conv_1x1'}, 'cell_3': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_3x3', 'normal_1': 'sep_conv_7x7'}, 'cell_4': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_3x3', 'normal_1': 'sep_conv_5x5'}}

arch_woDeconv_Img_worst5 = {'cell_0': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_7x7'}, 'cell_1': {'up_0': 'nearest', 'up_1': 'nearest', 'normal_0': 'conv_1x1', 'normal_1': 'sep_conv_7x7'}, 'cell_2': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_3x3', 'normal_1': 'conv_5x5'}, 'cell_3': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_5x5', 'normal_1': 'conv_3x3'}, 'cell_4': {'up_0': 'nearest', 'up_1': 'bilinear', 'normal_0': 'conv_3x3', 'normal_1': 'conv_5x5'}}

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    # config['skip_init'] = True
    config['skip_init'] = False    # because of flexible tasks, we need to initialize the network.
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator_retrain(**config, arch=arch_woDeconv_Img_worst5).to(device)
  D = model.Discriminator(**config).to(device)
 
  # Load D from pretrained models, and initialize the weights of embedding.
  #D.load_state_dict(torch.load(config['pretrain_path']))
  #for module in D.modules():
  #    if isinstance(module, nn.Embedding):
  #        init.orthogonal_(module.weight)

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator_retrain(**{**config, 'skip_init':True, 
                               'no_optim': True,
                               'arch': arch_woDeconv_Img_worst5}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  print(G)
  print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.parallel.DistributedDataParallel(
            GD,
            device_ids=[config['local_rank']],
            output_device=[config['local_rank']],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )


  
  if get_rank() == 0:
    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname, 
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname, 
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders_parallel(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  
  if get_rank() == 0:
    # Prepare tensorboard writer
    writer = SummaryWriter(os.path.join(config['logs_root'], experiment_name))
    global_steps = 0
    writer_dict = {'writer': writer,
                   'global_steps': global_steps}

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'], data_root=config['data_root'])

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  
  if get_rank() == 0:
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])  
    fixed_z.sample_()
    fixed_y.sample_()
  
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
                                            ema, state_dict, config, writer_dict)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  
  if get_rank() == 0:
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                  else G),
                               z_=z_, y_=y_, config=config)

  if get_rank() == 0:
    print('Beginning training at epoch %d...' % state_dict['epoch'])
    print(f'debug: length of data_loaders is {len(loaders[0])}')
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
  
  for i, (x, y) in enumerate(pbar):
    # Increment the iteration counter
    state_dict['itr'] += 1
    # Make sure G and D are in training mode, just in case they got set to eval
    # For D, which typically doesn't have BN, this shouldn't matter much.
    G.train()
    D.train()
    if config['ema']:
      G_ema.train()
    if config['D_fp16']:
      x, y = x.to(device).half(), y.to(device)
    else:
      x, y = x.to(device), y.to(device)
    metrics = train(x, y)
  
    if get_rank() == 0:  
      # Log the training process
      train_log.log(itr=int(state_dict['itr']), **metrics)
  
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
        print(', '.join(['itr: %d' % state_dict['itr']] 
                         + ['%s : %+4.3f' % (key, metrics[key])
                         for key in metrics]), end=' ')

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if not (state_dict['itr'] % config['test_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, experiment_name, test_log)


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()
