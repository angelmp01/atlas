"""
Atlas Preprocessing Script

This script provides a convenient way to run the data preprocessing pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing.data_preprocessor import DataPreprocessor


def run_preprocessing(input_dir=None, output_dir=None, config_path=None):
    """Run the preprocessing pipeline."""
    
    # Use default paths if not provided
    if input_dir is None:
        input_dir = project_root / "data" / "raw"
    
    if output_dir is None:
        output_dir = project_root / "data" / "processed"
    
    if config_path is None:
        config_path = project_root / "config" / "config.json"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_path}")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config_path=str(config_path) if config_path.exists() else None
        )
        
        # Run preprocessing
        output_file = preprocessor.run()
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📁 Output file: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return None


def main():
    """Main function."""
    print("🚛 Atlas Data Preprocessing Pipeline")
    print("=" * 40)
    
    # Run preprocessing
    print("Starting preprocessing...")
    output_file = run_preprocessing()
    
    if output_file:
        print(f"\n📊 Processing complete! Check the output file at:")
        print(f"   {output_file}")
        
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
    
    print("\n🎉 Atlas preprocessing pipeline ready for use!")


if __name__ == "__main__":
    main()
