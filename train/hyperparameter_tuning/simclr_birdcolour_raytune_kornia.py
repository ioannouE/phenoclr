import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import datetime as dt
import re
import yaml
import pytorch_lightning as pl
import torch
from copy import deepcopy
from datetime import datetime

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining

# from simclr_birdcolour_album import SimCLRModel, SimCLRDataModule
from lightning.pytorch.loggers import TensorBoardLogger

import socket 

def find_free_port():
    """Find a free port to use for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def train_simclr(config, base_config=None, album=False):

    # Set a unique port for each training process
    os.environ['MASTER_PORT'] = str(find_free_port())
    
    if args.album: # use albumentations for augmentations
        from simclr_birdcolour_kornia02 import SimCLRModel, SimCLRDataModule
    else:
        from simclr_birdcolour import SimCLRModel, SimCLRDataModule

    # Update base_config with tune parameters
    tune_config = deepcopy(base_config)

    if base_config['dataloader']['tune']:
        # Original batch size divided by number of GPUs
        per_gpu_batch_size = config['batch_size'] // torch.cuda.device_count()
        tune_config['dataloader']['batch_size'] = per_gpu_batch_size
    if base_config["model"]["criterion"]["params"]["tune"]:
        tune_config['model']['criterion']['params']['temperature'] = config['temperature']
    if base_config['train']['tune']:
        tune_config['train']['max_epochs'] = config['max_epochs']

    # Update augmentation parameters
    # tune_config['augmentations']['input_size'] = config['input_size']
    # tune_config['augmentations']['transforms']['random_resize_crop'].update({
    #     'scale': [config['random_resize_crop_scale_min'], config['random_resize_crop_scale_max']],
    #     'size': [config['input_size'], config['input_size']],
    #     'p': config['random_resize_crop_p']
    # })
    if album:
    
        if base_config['augmentations']['transforms']['random_resize_crop']['tune']:
            tune_config['augmentations']['transforms']['random_resize_crop'].update({
                'scale': [config['random_resize_crop_scale_min'], config['random_resize_crop_scale_max']],
                'size': [config['random_resize_crop_size'], config['random_resize_crop_size']],
                'p': 1.0
            })
        if base_config['augmentations']['transforms']['horizontal_flip']['tune']:
            tune_config['augmentations']['transforms']['horizontal_flip']['p'] = config['horizontal_flip_p']
        
        if base_config['augmentations']['transforms']['vertical_flip']['tune']:
            tune_config['augmentations']['transforms']['vertical_flip']['p'] = config['vertical_flip_p']
        
        if base_config['augmentations']['transforms']['rotate']['tune']:
            tune_config['augmentations']['transforms']['rotate'].update({
                # 'limit': config['rotate_limit'],
                'p': config['rotate_p']
            })
        if base_config['augmentations']['transforms']['gaussian_blur']['tune']:
            tune_config['augmentations']['transforms']['gaussian_blur'].update({
                # 'sigma_limit': [config['gaussian_blur_sigma_min'], config['gaussian_blur_sigma_max']],
                'p': config['gaussian_blur_p']
            })
        if base_config['augmentations']['transforms']['gaussian_noise']['tune']:
            tune_config['augmentations']['transforms']['gaussian_noise'].update({
                # 'std_range': [config['gaussian_noise_std_min'], config['gaussian_noise_std_max']],
                'p': config['gaussian_noise_p']
            })
        if base_config['augmentations']['transforms']['grayscale']['tune']:
            tune_config['augmentations']['transforms']['grayscale']['p'] = config['grayscale_p']
        
        if base_config['augmentations']['transforms']['thin_plate_spline']['tune']:
            tune_config['augmentations']['transforms']['thin_plate_spline']['p'] = config['thin_plate_spline_p']

        if base_config['augmentations']['transforms']['random_perspective']['tune']:
            tune_config['augmentations']['transforms']['random_perspective']['p'] = config['random_perspective_p']

        if base_config['augmentations']['transforms']['random_erasing']['tune']:
            tune_config['augmentations']['transforms']['random_erasing']['p'] = config['random_erasing_p']


    # Create data module with tune parameters
    data_module = SimCLRDataModule(tune_config)
    
    # Prepare model config to match the expected structure
    model_config = {
        'scheduler': tune_config['model']['scheduler'],
        'optimizer': tune_config['model']['optimizer'],
        'criterion': tune_config['model']['criterion'],
        'backbone': tune_config['model']['backbone'],
        'weights': tune_config['model']['weights']
    }
    
    model = SimCLRModel(model_config)
   
    
    # Get the trial directory for saving checkpoints
    trial_dir = session.get_trial_dir()
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss_ssl',  # Changed to match the actual logged metric
        dirpath=os.path.join(trial_dir, 'checkpoints'),
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
    )

    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=tune_config['train']['max_epochs'],
        accelerator=tune_config['train']['accelerator'],
        devices=tune_config['train']['devices'],
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[checkpoint_callback],
        deterministic=tune_config['train']['deterministic'],
        log_every_n_steps=tune_config['train']['log_every_n_steps'],
        strategy="ddp"
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Report final training loss
    final_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else float('inf')
    print(f"final_loss: {final_loss}")
    tune.report({'loss': final_loss})

def main(args):
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    
    
        
        # Define search space
        # search_space = {
        #     "batch_size": tune.choice([16, 32, 48]),
        #     "temperature": tune.uniform(0.05, 0.5),
        #     "max_epochs": tune.choice([100, 150, 200]),
        # }

    search_space = {}

    if base_config["dataloader"]["tune"]:
        search_space["batch_size"] = tune.choice(tuple(base_config["dataloader"]["batch_size"]))
                        
                            
    if base_config["model"]["criterion"]["params"]["tune"]:
        temp_range = base_config["model"]["criterion"]["params"]["temperature"]
        search_space["temperature"] = tune.uniform(temp_range[0], temp_range[1])                                

    if base_config["train"]["tune"]:
        search_space["max_epochs"] = tune.choice(tuple(base_config["train"]["max_epochs"]))


    if args.album:

        if base_config["augmentations"]["transforms"]["random_resize_crop"]["tune"]:
            size_range = base_config["augmentations"]["transforms"]["random_resize_crop"]["size"]
            scale_min_range = base_config["augmentations"]["transforms"]["random_resize_crop"]["scale_min"]
            scale_max_range = base_config["augmentations"]["transforms"]["random_resize_crop"]["scale_max"]
            search_space["random_resize_crop_scale_min"] = tune.uniform(scale_min_range[0], scale_min_range[1])
            search_space["random_resize_crop_scale_max"] = tune.uniform(scale_max_range[0], scale_max_range[1])
            search_space["random_resize_crop_size"] = tune.choice(tuple(size_range))

        if base_config["augmentations"]["transforms"]["grayscale"]["tune"]:
            grayscale_range = base_config["augmentations"]["transforms"]["grayscale"]["p"]
            search_space["grayscale_p"] = tune.uniform(grayscale_range[0], grayscale_range[1])
                                

        if base_config["augmentations"]["transforms"]["horizontal_flip"]["tune"]:
            horizontal_flip_range = base_config["augmentations"]["transforms"]["horizontal_flip"]["p"]
            search_space["horizontal_flip_p"] = tune.uniform(horizontal_flip_range[0], horizontal_flip_range[1])
                                

        if base_config["augmentations"]["transforms"]["vertical_flip"]["tune"]:
            vertical_flip_range = base_config["augmentations"]["transforms"]["vertical_flip"]["p"]
            search_space["vertical_flip_p"] = tune.uniform(vertical_flip_range[0], vertical_flip_range[1])
                                

        if base_config["augmentations"]["transforms"]["rotate"]["tune"]:
            # rotate_range = base_config["augmentations"]["transforms"]["rotate"]["limit"]
            # search_space["rotate_limit"] = tune.uniform(rotate_range[0], rotate_range[1])
            rotate_p = base_config["augmentations"]["transforms"]["rotate"]["p"]
            search_space["rotate_p"] = tune.uniform(rotate_p[0], rotate_p[1])
                                

        if base_config["augmentations"]["transforms"]["gaussian_blur"]["tune"]:
            # sigma_min_range = base_config["augmentations"]["transforms"]["gaussian_blur"]["sigma_min"]
            # sigma_max_range = base_config["augmentations"]["transforms"]["gaussian_blur"]["sigma_max"]
            # sigma_limit_range = base_config["augmentations"]["transforms"]["gaussian_blur"]["sigma_limit"]
            # search_space["gaussian_blur_sigma_min"] = tune.uniform(sigma_min_range[0], sigma_min_range[1])
            # search_space["gaussian_blur_sigma_max"] = tune.uniform(sigma_max_range[0], sigma_max_range[1])
            # search_space["gaussian_blur_sigma_limit"] = tune.uniform(sigma_limit_range[0], sigma_limit_range[1])
            
            gaussian_blur_range = base_config["augmentations"]["transforms"]["gaussian_blur"]["p"]
            search_space["gaussian_blur_p"] = tune.uniform(gaussian_blur_range[0], gaussian_blur_range[1])


        if base_config["augmentations"]["transforms"]["gaussian_noise"]["tune"]:
            gaussian_noise_std_min = base_config["augmentations"]["transforms"]["gaussian_noise"]["std_min"]
            gaussian_noise_std_max = base_config["augmentations"]["transforms"]["gaussian_noise"]["std_max"]
            search_space["gaussian_noise_std_min"] = tune.uniform(gaussian_noise_std_min[0], gaussian_noise_std_min[1])
            search_space["gaussian_noise_std_max"] = tune.uniform(gaussian_noise_std_max[0], gaussian_noise_std_max[1])

            gaussian_noise_range = base_config["augmentations"]["transforms"]["gaussian_noise"]["p"]
            search_space["gaussian_noise_p"] = tune.uniform(gaussian_noise_range[0], gaussian_noise_range[1])
        
        if base_config["augmentations"]["transforms"]["thin_plate_spline"]["tune"]:
            thin_plate_spline_range = base_config["augmentations"]["transforms"]["thin_plate_spline"]["p"]
            search_space["thin_plate_spline_p"] = tune.uniform(thin_plate_spline_range[0], thin_plate_spline_range[1])


        if base_config["augmentations"]["transforms"]["random_perspective"]["tune"]:
            random_perspective_range = base_config["augmentations"]["transforms"]["random_perspective"]["p"]
            search_space["random_perspective_p"] = tune.uniform(random_perspective_range[0], random_perspective_range[1])


        if base_config["augmentations"]["transforms"]["random_erasing"]["tune"]:
            random_erasing_range = base_config["augmentations"]["transforms"]["random_erasing"]["p"]
            search_space["random_erasing_p"] = tune.uniform(random_erasing_range[0], random_erasing_range[1])


    # Add this before ray.init()
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
    ray.init(num_gpus=args.num_gpus)
    
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations=search_space
    )
  
    
    # Ensure output directory is absolute path
    if not os.path.isabs(base_config['output']['out_dir']):
        base_config['output']['out_dir'] = os.path.abspath(base_config['output']['out_dir'])
    os.makedirs(base_config['output']['out_dir'], exist_ok=True)
    
    # Run hyperparameter optimization
    analysis = tune.run(
        tune.with_parameters(train_simclr, base_config=base_config, album=args.album),
        config=search_space,
        num_samples=args.num_samples,
        scheduler=pbt_scheduler,
        storage_path=os.path.join(base_config['output']['out_dir'], 'ray_results'),
        name=f"simclr_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        resources_per_trial={
            "cpu": 16,
            "gpu": args.num_gpus / args.num_parallel
        }
    )
    
    # Print and save best config
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best trial config:", best_trial.config)
    print("Best trial final loss:", best_trial.last_result["loss"])
    
    best_config_path = os.path.join(base_config['output']['out_dir'], 'best_config.yaml')
    with open(best_config_path, 'w') as f:
        yaml.dump(best_trial.config, f)
    print(f"Best config saved to: {best_config_path}")


def get_args():
    parser = argparse.ArgumentParser(description='SimCLR with Ray Tune')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to base config file')
    parser.add_argument('--num-samples', type=int, default=10,
                      help='Number of trials to run')
    parser.add_argument('--num-gpus', type=int, default=1,
                      help='Number of GPUs to use')
    parser.add_argument('--num-parallel', type=int, default=1,
                      help='Number of parallel trials')
    parser.add_argument('--album', action='store_true',
                      help='Optimize augmentations (kornia) hyperparameters')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)