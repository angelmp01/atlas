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
    parser = argparse.ArgumentParser(description='Atlas Data Preprocessing - Hermes Trip Expansion')
    parser.add_argument('--input-dir', 
                       default=str(project_root / "data" / "raw"),
                       help='Directory containing raw data files')
    parser.add_argument('--output-dir', 
                       default=str(project_root / "data" / "processed"),
                       help='Directory to save processed CSV file')
    parser.add_argument('--config', 
                       default=str(project_root / "config" / "config.json"),
                       help='Path to configuration file')
    parser.add_argument('--objectid', 
                       type=int,
                       help='Process only this OBJECTID for validation (optional)')
    parser.add_argument('--distribution', 
                       choices=['exact', 'poisson', 'normal'],
                       default='exact',
                       help='Distribution mode for trip generation: exact (default), poisson, or normal')
    parser.add_argument('--seed', 
                       type=int,
                       help='Random seed for reproducible results when using probabilistic distributions')
    
    args = parser.parse_args()
    
    print("🚛 Atlas Data Preprocessing Pipeline - Hermes Trip Expansion")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.config}")
    print(f"Distribution mode: {args.distribution}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    if args.objectid:
        print(f"🔍 Validation mode: Processing only OBJECTID {args.objectid}")
    
    try:
        # Initialize preprocessor
        config_path = args.config if Path(args.config).exists() else None
        preprocessor = DataPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=config_path,
            target_objectid=args.objectid,
            distribution_mode=args.distribution,
            random_seed=args.seed
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
            print(f"   • Total trip records: {len(df)}")
            print(f"   • Columns: {len(df.columns)}")
            print(f"   • Data sources: {df['source_file'].nunique() if 'source_file' in df.columns else 'Unknown'}")
            
            # Additional statistics for expanded trip data
            if 'objectid' in df.columns:
                print(f"   • Unique routes (OBJECTID): {df['objectid'].nunique()}")
            if 'fecha' in df.columns:
                print(f"   • Date range: {df['fecha'].min()} to {df['fecha'].max()}")
            if 'dia_semana' in df.columns:
                day_counts = df['dia_semana'].value_counts()
                print(f"   • Trips by day: {dict(day_counts)}")
            if 'modo_distribucion' in df.columns:
                distribution_mode = df['modo_distribucion'].iloc[0]
                print(f"   • Distribution mode used: {distribution_mode}")
                
                # Show distribution statistics if not exact
                if distribution_mode != 'exact' and 'promedio_viajes_dia' in df.columns:
                    avg_trips = df.groupby('dia_semana').agg({
                        'promedio_viajes_dia': 'first',
                        'total_viajes_dia': 'mean'
                    }).round(2)
                    print(f"   • Average vs Generated trips by day:")
                    for day, row in avg_trips.iterrows():
                        print(f"     {day}: {row['promedio_viajes_dia']} avg → {row['total_viajes_dia']} generated")
            
        except Exception as e:
            print(f"Could not read output file for summary: {e}")
        
        print(f"\n🎉 Hermes trip expansion completed successfully!")
        if args.objectid:
            print(f"🔍 Validation completed for OBJECTID {args.objectid}")
        return output_file
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return None


if __name__ == "__main__":
    main()
