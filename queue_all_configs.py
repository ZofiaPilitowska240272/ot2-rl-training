"""
Simple Script to Queue All 12 Configs to ClearML
-------------------------------------------------
No interactive prompts - just runs all 12 configs.

Author: Zofia Pilitowska (240272)

Usage:
    python queue_all_configs.py
"""

import os
from clearml import Task
from configs_12_models import CONFIGS
import time


def queue_experiment(config_index):
    """Queue a single experiment to ClearML."""
    
    config = CONFIGS[config_index]
    
    print(f"\n{'='*70}")
    print(f"[{config_index + 1}/12] {config['name']}")
    print(f"{'='*70}")
    
    # Create task using Task.create (not Task.init!)
    task = Task.create(
        project_name='Mentor Group - Uther/Group 2',
        task_name=config['name'],
        task_type=Task.TaskTypes.training,
        script='train_rl.py',
        working_directory='.',
        repo=None,
        branch=None,
        add_task_init_call=False  # Changed to False - train_rl.py already has Task.init
    )
    
    # Set execution settings
    task.set_base_docker('deanis/2023y2b-rl:latest')
    
    # IMPORTANT: FORCE install critical packages
    # Always add these explicitly to ensure they're installed
    critical_packages = [
        'clearml==2.0.2',
        'wandb==0.23.1',
    ]
    
    # Then add requirements from file
    requirements_files = ['requirements.txt', 'requirements_clearml.txt']
    packages_set = False
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"  Using requirements from: {req_file}")
            with open(req_file, 'r') as f:
                requirements = [line.strip() for line in f 
                              if line.strip() and not line.startswith('#')]
            
            # Combine: critical packages FIRST, then requirements
            all_packages = critical_packages + requirements
            task.set_packages(all_packages)
            packages_set = True
            print(f"  Total packages: {len(all_packages)} (including {len(critical_packages)} critical)")
            break
    
    if not packages_set:
        # Fallback: Just critical + essentials
        print("  Using manual package list")
        task.set_packages(critical_packages + [
            'stable-baselines3==2.7.0',
            'gymnasium==1.2.2',
            'pybullet==3.2.7',
            'torch==2.9.1',
            'numpy==1.26.2',
            'matplotlib==3.10.7',
            'pandas==2.1.4'
        ])
    
    # Build command-line arguments for train_rl.py
    args_dict = {
        'algorithm': config['algorithm'],
        'total_timesteps': config['total_timesteps'],
        'n_envs': config['n_envs'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'n_steps': config['n_steps'],
        'net_arch': config['net_arch'],
        'gamma': config['gamma'],
        'gae_lambda': config['gae_lambda'],
        'clip_range': config['clip_range'],
        'ent_coef': config['ent_coef'],
        'vf_coef': config['vf_coef'],
        'max_velocity': config['max_velocity'],
        'reward_scale': config['reward_scale'],
        'max_steps': config['max_steps'],
        'success_threshold': 0.005,
        'seed': config['seed'],
        'run_name': config['name'],
        'use_gpu': True
    }
    
    # Connect parameters to task
    task.connect(args_dict)
    
    # Enqueue to default queue
    Task.enqueue(task, queue_name='default')
    
    print(f"✓ Queued to ClearML")
    print(f"  Task ID: {task.id}")
    print(f"  LR: {config['learning_rate']:.2e} | Network: {config['net_arch']}")
    print(f"  Velocity: {config['max_velocity']} | Steps: {config['max_steps']}")
    
    return task.id


def main():
    print("\n" + "="*70)
    print("QUEUEING ALL 12 CONFIGS TO CLEARML")
    print("="*70)
    print("Project: Mentor Group - Uther/Group 2")
    print("Queue: default")
    print("Docker: deanis/2023y2b-rl:latest")
    print("="*70)
    
    task_ids = []
    
    # Queue all 12 configs
    for i in range(12):
        try:
            task_id = queue_experiment(i)
            task_ids.append((i, CONFIGS[i]['name'], task_id))
            time.sleep(1)  # Small delay between submissions
        except Exception as e:
            print(f"❌ Error queueing config {i}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("ALL CONFIGS QUEUED!")
    print("="*70)
    print(f"Total queued: {len(task_ids)}/12")
    print("\nTask Summary:")
    for idx, name, task_id in task_ids:
        print(f"  {idx}. {name}")
        print(f"     {task_id}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Go to ClearML UI: https://app.clear.ml")
    print("2. Navigate to 'Mentor Group - Uther/Group 2' project")
    print("3. Monitor queue and experiments")
    print("4. Workers will pick up tasks automatically")
    print("\nEstimated total time: ~24-48 hours for all 12 configs")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()