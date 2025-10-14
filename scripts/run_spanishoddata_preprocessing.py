"""
Script to run SpanishODData preprocessing with synthetic cargo fields.

This script provides a convenient way to run the SpanishODData preprocessor
from the project root directory.
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add src to Python path
sys.path.insert(0, str(project_root / "src"))

from preprocessing.spanishoddata_main import main


def run_spanishoddata_preprocessing(input_dir=None, output_dir=None, config_path=None, seed=None):
    """
    Run SpanishODData preprocessing with optional parameters.
    
    Args:
        input_dir (str, optional): Directory containing SpanishODData CSV files
        output_dir (str, optional): Output directory for processed data
        config_path (str, optional): Path to configuration file
        seed (int, optional): Random seed for reproducibility
    """
    
    # Set defaults if not provided
    if input_dir is None:
        input_dir = str(project_root / "data" / "raw" / "spanishoddata")
    
    if output_dir is None:
        output_dir = str(project_root / "data" / "processed")
    
    if config_path is None:
        config_path = str(project_root / "config" / "config.json")
    
    # Override sys.argv to pass arguments to main()
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = ['spanishoddata_main.py']
        sys.argv.extend(['--input-dir', input_dir])
        sys.argv.extend(['--output-dir', output_dir])
        sys.argv.extend(['--config', config_path])
        
        if seed is not None:
            sys.argv.extend(['--seed', str(seed)])
        
        # Run the main function
        result = main()
        return result
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    # If run directly, call the main function with sys.argv
    result = main()
    if result:
        print(f"\nScript completed successfully. Output: {result}")
    else:
        print("\nScript failed.")
        sys.exit(1)