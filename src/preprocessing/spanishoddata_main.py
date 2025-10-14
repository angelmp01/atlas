"""
Main entry point for the Atlas SpanishODData preprocessing module.

This script handles the preprocessing of SpanishODData CSV files by adding
synthetic cargo fields without expanding rows.
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd

# Get the project root (parent of src)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing.spanishoddata_preprocessor import SpanishODDataPreprocessor


def main():
    """Main function for SpanishODData preprocessing entry point."""
    parser = argparse.ArgumentParser(description='Atlas SpanishODData Preprocessing - Synthetic Cargo Fields')
    parser.add_argument('--input-dir', 
                       default=str(project_root / "data" / "raw" / "spanishoddata"),
                       help='Directory containing SpanishODData CSV files')
    parser.add_argument('--output-dir', 
                       default=str(project_root / "data" / "processed"),
                       help='Directory to save processed CSV files')
    parser.add_argument('--config', 
                       default=str(project_root / "config" / "config.json"),
                       help='Path to configuration file')
    parser.add_argument('--seed', 
                       type=int,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    print("📊 Atlas SpanishODData Preprocessing Pipeline - Synthetic Cargo Fields")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {args.config}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    
    try:
        # Initialize preprocessor
        config_path = args.config if Path(args.config).exists() else None
        preprocessor = SpanishODDataPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=config_path,
            random_seed=args.seed
        )
        
        # Run preprocessing
        output_files = preprocessor.run()
        
        print(f"\n✅ SpanishODData preprocessing completed successfully!")
        print(f"📁 Generated {len(output_files)} output files:")
        for output_file in output_files:
            print(f"   • {output_file}")
        
        # Show summary for all files
        total_records = 0
        total_trips = 0
        total_distance = 0
        
        for output_file in output_files:
            try:
                df = pd.read_csv(output_file)
                total_records += len(df)
                if 'n_trips' in df.columns:
                    total_trips += df['n_trips'].sum()
                if 'trips_total_length_km' in df.columns:
                    total_distance += df['trips_total_length_km'].sum()
                    
                print(f"\n📈 Summary for {Path(output_file).name}:")
                print(f"   • Records: {len(df)}")
                if 'n_trips' in df.columns:
                    print(f"   • Trips: {df['n_trips'].sum():,}")
                if 'trips_total_length_km' in df.columns:
                    print(f"   • Distance: {df['trips_total_length_km'].sum():,.0f} km")
                
                # Show synthetic field distributions for first file only (to avoid spam)
                if output_file == output_files[0]:
                    if 'tipo_mercancia' in df.columns:
                        mercancia_counts = df['tipo_mercancia'].value_counts()
                        print(f"   • Cargo types: {dict(mercancia_counts)}")
                    
                    if 'volumen' in df.columns:
                        vol_stats = df['volumen'].describe()
                        print(f"   • Volume range: {vol_stats['min']:.0f} - {vol_stats['max']:.0f} palets (avg: {vol_stats['mean']:.1f})")
                    
                    if 'peso' in df.columns:
                        peso_stats = df['peso'].describe()
                        print(f"   • Weight range: {peso_stats['min']:.0f} - {peso_stats['max']:.0f} kg (avg: {peso_stats['mean']:.0f})")
                    
                    if 'precio' in df.columns:
                        precio_stats = df['precio'].describe()
                        print(f"   • Price range: {precio_stats['min']:.0f} - {precio_stats['max']:.0f} € (avg: {precio_stats['mean']:.0f})")
                        
            except Exception as e:
                print(f"Could not read output file {output_file} for summary: {e}")
        
        print(f"\n🎯 Overall Summary:")
        print(f"   • Total files processed: {len(output_files)}")
        print(f"   • Total records: {total_records:,}")
        print(f"   • Total trips represented: {total_trips:,}")
        print(f"   • Total distance: {total_distance:,.0f} km")
        
        print(f"\n🎉 SpanishODData synthetic field generation completed successfully!")
        return output_files
        
    except Exception as e:
        print(f"\n❌ SpanishODData preprocessing failed: {e}")
        return None


if __name__ == "__main__":
    main()