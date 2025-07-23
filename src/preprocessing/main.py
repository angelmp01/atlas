"""
Main entry point for the Atlas preprocessing module.

This script handles the preprocessing of raw transportation and delivery data.
"""

import sys
import os
from pathlib import Path
import argparse

# Get the project root (parent of src)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing.data_preprocessor import DataPreprocessor


def main():
    """Main function for preprocessing entry point."""
    parser = argparse.ArgumentParser(description='Atlas Data Preprocessing')
    parser.add_argument('--input-dir', 
                       default=str(project_root / "data" / "raw"),
                       help='Directory containing raw data files')
    parser.add_argument('--output-dir', 
                       default=str(project_root / "data" / "processed"),
                       help='Directory to save processed CSV file')
    parser.add_argument('--config', 
                       default=str(project_root / "config" / "config.json"),
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("🚛 Atlas Data Preprocessing Pipeline")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.config}")
    
    try:
        # Initialize preprocessor
        config_path = args.config if Path(args.config).exists() else None
        preprocessor = DataPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=config_path
        )
        
        # Run preprocessing
        output_file = preprocessor.run()
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📁 Output file: {output_file}")
        
        # Show basic stats
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"\n📈 Data Summary:")
            print(f"   • Total deliveries: {len(df)}")
            print(f"   • Columns: {len(df.columns)}")
            print(f"   • Data sources: {df['source_file'].nunique() if 'source_file' in df.columns else 'Unknown'}")
            
        except Exception as e:
            print(f"Could not read output file for summary: {e}")
        
        print("\n🎉 Preprocessing completed successfully!")
        return output_file
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return None


if __name__ == "__main__":
    main()
