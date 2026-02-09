import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set the seed for Python, NumPy, and PyTorch (CPU and GPU) for reproducible results.
    
    Ensures deterministic behavior across random number generators by setting seeds for
    Python's random module, NumPy, and PyTorch (both CPU and GPU). Also configures
    CuDNN for deterministic operations.
    
    Args:
        seed (int): Random seed value. Defaults to 42.
        
    Returns:
        int: The seed value that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CuDNN behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def unset_seeds():
    """
    Reset random number generators to non-deterministic behavior using time-based seeds.
    
    Resets Python's random module, NumPy, and PyTorch to use time-based random seeds,
    allowing for non-reproducible random behavior. Also re-enables non-deterministic
    CuDNN operations.
    
    Returns:
        None
    """
    # Get current time-based seed
    import time
    current_time_seed = int(time.time() * 1000) % (2**32)
    
    # Reset with time-based seeds
    random.seed(current_time_seed)
    np.random.seed(current_time_seed % (2**32))
    torch.manual_seed(current_time_seed % (2**63))  # PyTorch uses signed 64-bit
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(current_time_seed % (2**63))
        torch.cuda.manual_seed_all(current_time_seed % (2**63))
    
    # Re-enable non-deterministic behavior
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def reset_to_random():
    """
    Alternative method to reset random number generators to completely random state.
    
    Uses system entropy and random seed generation to reset NumPy and PyTorch
    random states, allowing libraries to use their default random initialization.
    
    Returns:
        None
    """
    # Don't set any seed - let libraries use their default random initialization
    
    # For NumPy, you can reset the random state
    np.random.seed(None)  # Uses system time/entropy
    
    # For PyTorch, generate a random seed
    random_seed = torch.randint(0, 2**31, (1,)).item()
    torch.manual_seed(random_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
