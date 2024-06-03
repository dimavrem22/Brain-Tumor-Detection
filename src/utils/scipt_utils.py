import json
import subprocess
import torch
from dataclasses import dataclass, field


def get_device():
    """
    Get the device to use for training (cuda if available, then mps, then cpu)
    """
    if torch.cuda.is_available():
        print("Running on CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Running on MPS")
        return torch.device("mps")
    else:
        print("Running on CPU")
        return torch.device("cpu")


def get_latest_commit_id():

    process = subprocess.Popen(['git', '-C', ".", 'rev-parse', 'HEAD'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        # Handle error
        print(f"Error: {stderr.decode().strip()}")
        return None
    
    # Return the latest commit ID
    return stdout.decode().strip()


def save_configs_dict(configs_dict, save_path, add_git_commit_id=True):

    configs_dict_copy = configs_dict.copy()

    if add_git_commit_id:
        configs_dict_copy['git commit id'] = get_latest_commit_id()

    for key, value in configs_dict_copy.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            configs_dict_copy[key] = str(value)
        
    
    with open(save_path, "w") as f:
        json.dump(configs_dict_copy, f, indent=4)