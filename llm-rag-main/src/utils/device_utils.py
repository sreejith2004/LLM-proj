"""
Device detection and configuration utilities.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def get_device_info():
    """Get information about available devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        'device_count': 0,
        'device_name': 'CPU',
        'recommended_dtype': torch.float32,
        'recommended_precision': 'fp32'
    }

    if info['cuda_available']:
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = f"CUDA ({torch.cuda.get_device_name(0)})"
        info['recommended_dtype'] = torch.float16
        info['recommended_precision'] = 'fp16'

    elif info['mps_available']:
        info['device_name'] = "Apple Silicon (MPS)"
        info['recommended_dtype'] = torch.float32
        info['recommended_precision'] = 'fp32'

    return info

def get_optimal_training_config():
    """Get optimal training configuration based on device."""
    device_info = get_device_info()

    config = {
        'fp16': False,
        'bf16': False,
        'dataloader_pin_memory': False,
        'use_quantization': False
    }

    if device_info['cuda_available']:
        config['fp16'] = True
        config['dataloader_pin_memory'] = True
        config['use_quantization'] = True

    elif device_info['mps_available']:
        # Use fp32 for MPS compatibility
        pass

    return config

def log_device_info():
    """Log device information."""
    info = get_device_info()

    logger.info("=" * 50)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Device: {info['device_name']}")
    logger.info(f"CUDA Available: {info['cuda_available']}")
    logger.info(f"MPS Available: {info['mps_available']}")
    logger.info(f"Recommended dtype: {info['recommended_dtype']}")
    logger.info(f"Recommended precision: {info['recommended_precision']}")

    if info['cuda_available']:
        logger.info(f"CUDA Device Count: {info['device_count']}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("=" * 50)

def main():
    """Test device detection."""
    log_device_info()

    config = get_optimal_training_config()
    print("\nOptimal Training Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
