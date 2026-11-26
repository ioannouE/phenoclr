import os
import pytorch_lightning as pl
from sklearn.model_selection import ParameterGrid
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import json
from datetime import datetime
from argparse import ArgumentParser
from copy import deepcopy

from simclr_birdcolour_kornia import SimCLRModel, SimCLRDataModule

def extract_grid_parameters(config):
    """Extract parameters for grid search from config"""
    param_grid = {}
    
    # Extract batch size if it's a list
    if isinstance(config.get('dataloader', {}).get('batch_size'), list):
        param_grid['batch_size'] = config['dataloader']['batch_size']
    
    # Extract max epochs if it's a list
    if isinstance(config.get('train', {}).get('max_epochs'), list):
        param_grid['max_epochs'] = config['train']['max_epochs']
    
    # Extract temperature if it's a list
    if isinstance(config.get('model', {}).get('criterion', {}).get('params', {}).get('temperature'), list):
        param_grid['temperature'] = config['model']['criterion']['params']['temperature']
    
    if not param_grid:
        raise ValueError("No parameters for grid search found in config file. "
                        "Make sure at least one of batch_size, max_epochs, or temperature is a list.")
    
    return param_grid

def update_config_with_params(config, params):
    """Update configuration with grid search parameters"""
    new_config = deepcopy(config)
    
    # Update only the parameters that were part of the grid search
    if 'batch_size' in params:
        new_config['dataloader']['batch_size'] = params['batch_size']
    if 'max_epochs' in params:
        new_config['train']['max_epochs'] = params['max_epochs']
    if 'temperature' in params:
        new_config['model']['criterion']['params']['temperature'] = params['temperature']
    
    # Ensure all required nested structures exist
    if 'model' not in new_config:
        new_config['model'] = {}
    if 'scheduler' not in new_config['model']:
        new_config['model']['scheduler'] = {}
    if 'params' not in new_config['model']['scheduler']:
        new_config['model']['scheduler']['params'] = {}
    if 'optimizer' not in new_config['model']:
        new_config['model']['optimizer'] = {}
    
    return new_config

def train_and_evaluate(config, run_dir):
    """Train and evaluate a single configuration"""
    # Initialize data module and model
    data_module = SimCLRDataModule(config)
    
    # Prepare model config to match the expected structure
    model_config = {
        'scheduler': config['model']['scheduler'],
        'optimizer': config['model']['optimizer'],
        'criterion': config['model']['criterion'],
        'backbone': config['model']['backbone'],
        'weights': config['model']['weights']
    }
    
    model = SimCLRModel(model_config)
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir=run_dir,
        name=f"bs_{config['dataloader']['batch_size']}_ep_{config['train']['max_epochs']}_t_{config['model']['criterion']['params']['temperature']}"
    )
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_ssl',  # Changed to match the actual logged metric
        dirpath=os.path.join(run_dir, 'checkpoints'),
        filename='{epoch:02d}-{train_loss_ssl:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    # Initialize trainer with all config parameters
    trainer = pl.Trainer(
        max_epochs=config['train']['max_epochs'],
        accelerator=config['train']['accelerator'],
        devices=config['train']['devices'],
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=config['train']['deterministic'],
        log_every_n_steps=config['train']['log_every_n_steps']
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    return {
        'best_loss': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else float('inf'),
        'best_model_path': checkpoint_callback.best_model_path
    }

def main(args):
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Extract parameter grid from config
    param_grid = extract_grid_parameters(base_config)
    
    print("Parameters being optimized:", list(param_grid.keys()))
    for param, values in param_grid.items():
        print(f"{param}: {values}")
    
    # Create grid of all parameter combinations
    grid = list(ParameterGrid(param_grid))
    print(f"\nTotal number of configurations to try: {len(grid)}")
    
    # If param_i is specified, only run that configuration
    if args.param_i >= 0:
        if args.param_i >= len(grid):
            raise ValueError(f"param_i ({args.param_i}) must be less than total configurations ({len(grid)})")
        grid = [grid[args.param_i]]
        print(f"Running only configuration {args.param_i}")
    
    # Create directory for grid search results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_search_dir = os.path.join(base_config['output']['out_dir'], f'grid_search_{timestamp}')
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Save grid search metadata
    metadata = {
        'total_configurations': len(grid),
        'parameters_optimized': list(param_grid.keys()),
        'parameter_values': param_grid
    }
    with open(os.path.join(grid_search_dir, 'grid_search_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results = []
    
    # Run training for each configuration
    for i, params in enumerate(grid):
        print(f"\nTraining configuration {i+1}/{len(grid)}")
        print("Parameters:", params)
        
        # Update config with grid search parameters
        current_config = update_config_with_params(base_config, params)
        
        # Create run directory
        run_dir = os.path.join(grid_search_dir, f'run_{i+1}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Save current configuration
        with open(os.path.join(run_dir, 'config_grid.yaml'), 'w') as f:
            yaml.dump(current_config, f)
        
        try:
            # Train and evaluate
            train_results = train_and_evaluate(current_config, run_dir)
            
            # Store results
            result = {
                'run_id': i+1,
                'parameters': params,
                'best_loss': train_results['best_loss'],
                'best_model_path': train_results['best_model_path']
            }
            results.append(result)
            
            print(f"Run {i+1} completed. Best loss: {train_results['best_loss']:.4f}")
            
            # Save results after each run (in case of crashes)
            with open(os.path.join(grid_search_dir, 'grid_search_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Error in run {i+1}: {str(e)}")
            continue
    
    # Print best configuration
    if results:
        best_run = min(results, key=lambda x: x['best_loss'])
        print("\nBest configuration found:")
        print(json.dumps(best_run, indent=2))

def get_args():
    parser = ArgumentParser(description='SimCLR Grid Search')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the base config.yaml file')
    parser.add_argument('--param_i', type=int, default=-1,
                      help='Index of parameter combination to run (-1 for all)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)