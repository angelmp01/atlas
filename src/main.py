"""
Atlas Main Entry Point

This script provides a unified interface to all Atlas modules.
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    """Main entry point for Atlas project."""
    parser = argparse.ArgumentParser(
        description='Atlas - Data Processing for Truck Delivery Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modules:
  preprocessing     Process raw delivery data into standardized CSV format

Examples:
  python src/main.py preprocessing --help
  python src/main.py preprocessing --input-dir data/raw --output-dir data/processed
        """
    )
    
    parser.add_argument('module', 
                       choices=['preprocessing'],
                       help='Module to run')
    parser.add_argument('--version', 
                       action='version', 
                       version='Atlas 0.1.0')
    
    # Parse only the module argument first
    args, remaining = parser.parse_known_args()
    
    print("🚛 Atlas - Data Processing for Truck Delivery Analytics")
    print("=" * 50)
    
    if args.module == 'preprocessing':
        from preprocessing.main import main as preprocessing_main
        # Temporarily modify sys.argv to pass remaining arguments to the module
        original_argv = sys.argv
        sys.argv = ['preprocessing'] + remaining
        try:
            result = preprocessing_main()
            sys.argv = original_argv
            return result
        except SystemExit:
            sys.argv = original_argv
            return None


if __name__ == "__main__":
    main()
