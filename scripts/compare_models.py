"""
Compare models and visualize performance.

Reads model_card.json from multiple models and compares them.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_model_metrics(model_path: Path) -> Dict[str, Any]:
    """Load metrics from a model's model_card.json."""
    
    model_card_path = model_path / "model_card.json"
    
    if not model_card_path.exists():
        logger.warning(f"No model_card.json found in {model_path}")
        return None
    
    with open(model_card_path, 'r') as f:
        model_card = json.load(f)
    
    cv_metrics = model_card['performance']['cv_metrics']
    overall = model_card['performance']['overall_metrics']
    
    # Calculate model size in MB
    model_size_mb = 0
    model_file = model_path / "model.joblib"
    encoders_file = model_path / "encoders.joblib"
    
    if model_file.exists():
        model_size_mb += model_file.stat().st_size / (1024 * 1024)
    if encoders_file.exists():
        model_size_mb += encoders_file.stat().st_size / (1024 * 1024)
    
    return {
        'model_name': model_path.name,
        'model_path': model_path,
        'created_at': model_card['created_at'],
        'task_type': model_card['task_type'],
        'model_type': model_card['model_type'],
        'n_features': model_card['training_config']['features']['count'],
        'n_folds': model_card['training_config']['cv_setup']['n_folds'],
        'training_samples': model_card.get('additional', {}).get('training_samples', 'N/A'),
        'size_mb': model_size_mb,
        # CV metrics
        'mae_mean': cv_metrics['mae_mean'],
        'mae_std': cv_metrics['mae_std'],
        'rmse_mean': cv_metrics['rmse_mean'],
        'r2_mean': float(cv_metrics['r2_mean']),
        # Overall metrics
        'mae_overall': overall['mae'],
        'rmse_overall': overall['rmse'],
        'r2_overall': float(overall['r2']),
    }


def compare_models(models_dir: str = "models", task_type: str = None):
    """
    Compare all models in the models directory.
    
    Args:
        models_dir: Directory containing model bundles
        task_type: Filter by task type ('probability', 'price', 'weight')
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*80}\n")
    
    # Load all models
    model_metrics = []
    
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            metrics = load_model_metrics(model_dir)
            if metrics:
                # Filter by task type if specified
                if task_type is None or metrics['task_type'] == task_type:
                    model_metrics.append(metrics)
    
    if not model_metrics:
        logger.warning("No models found")
        return
    
    # Create DataFrame for easy comparison
    df = pd.DataFrame(model_metrics)
    
    # Sort by MAE (lower is better)
    df = df.sort_values('mae_mean')
    
    logger.info(f"Found {len(df)} models total")
    if task_type:
        logger.info(f"Filtered by task_type: {task_type}")
    
    # Group by task type and display as tables
    for task in sorted(df['task_type'].unique()):
        task_df = df[df['task_type'] == task].copy()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"{task.upper()} MODELS")
        logger.info(f"{'='*80}\n")
        
        # Prepare table data
        table_data = []
        for idx, row in task_df.iterrows():
            # Format date as DD/MM/YYYY HH:MM
            created_dt = datetime.fromisoformat(row['created_at'])
            date_str = created_dt.strftime("%d/%m/%Y %H:%M")
            
            table_data.append([
                row['model_name'][:35],  # Truncate long names
                date_str,
                f"{row['size_mb']:.1f}",
                f"{row['mae_mean']:.6f}",
                f"{row['mae_std']:.6f}",
                f"{row['r2_mean']:.6f}",
                f"{row['rmse_mean']:.6f}",
                row['n_features']
            ])
        
        # Add separator and best model indicator
        if len(table_data) > 0:
            best = task_df.iloc[0]
            created_dt = datetime.fromisoformat(best['created_at'])
            date_str = created_dt.strftime("%d/%m/%Y %H:%M")
            
            table_data.append(['─' * 35, '─' * 16, '─' * 6, '─' * 10, '─' * 10, '─' * 10, '─' * 10, '─' * 8])
            table_data.append([
                f"✓ {best['model_name'][:32]}",
                date_str,
                f"{best['size_mb']:.1f}",
                f"{best['mae_mean']:.6f}",
                f"{best['mae_std']:.6f}",
                f"{best['r2_mean']:.6f}",
                f"{best['rmse_mean']:.6f}",
                best['n_features']
            ])
        
        headers = ['Model', 'Date', 'Size MB', 'MAE', 'MAE_Std', 'R²', 'RMSE', 'Features']
        
        table_str = tabulate(
            table_data,
            headers=headers,
            tablefmt='simple',
            stralign='left'
        )
        
        logger.info(table_str)
        logger.info("")
    
    # Summary table with best models
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: BEST MODEL PER TASK")
    logger.info(f"{'='*80}\n")
    
    summary_data = []
    for task in sorted(df['task_type'].unique()):
        task_df = df[df['task_type'] == task]
        best = task_df.iloc[0]
        created_dt = datetime.fromisoformat(best['created_at'])
        date_str = created_dt.strftime("%d/%m/%Y %H:%M")
        
        summary_data.append([
            task.upper(),
            best['model_name'],
            date_str,
            f"{best['size_mb']:.1f} MB",
            f"{best['mae_mean']:.6f} ± {best['mae_std']:.6f}",
            f"{best['r2_mean']:.6f}"
        ])
    
    summary_table = tabulate(
        summary_data,
        headers=['Task', 'Best Model', 'Date', 'Size', 'MAE (CV)', 'R²'],
        tablefmt='simple',
        stralign='left'
    )
    
    logger.info(summary_table)
    logger.info("")
    
    # Add interpretation help
    logger.info("INTERPRETATION:")
    logger.info("  • MAE (Mean Absolute Error): Average prediction error (lower is better)")
    logger.info("  • R² (R-squared): Variance explained by model, 0-1 scale (higher is better)")
    logger.info("  • RMSE (Root Mean Squared Error): Penalizes large errors (lower is better)")
    logger.info("  • ✓ indicates the best model (lowest MAE) for each task")
    logger.info("")
    
    # Save comparison
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV (remove model_path column which is not serializable)
    df_to_save = df.drop(columns=['model_path'], errors='ignore')
    csv_file = output_dir / f"model_comparison_{timestamp}.csv"
    df_to_save.to_csv(csv_file, index=False)
    logger.info(f"Comparison saved to: {csv_file}")
    
    # Save as JSON (convert Path objects to strings)
    json_file = output_dir / f"model_comparison_{timestamp}.json"
    model_metrics_json = []
    for m in model_metrics:
        m_copy = m.copy()
        if 'model_path' in m_copy:
            m_copy['model_path'] = str(m_copy['model_path'])
        model_metrics_json.append(m_copy)
    
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': model_metrics_json
        }, f, indent=2)
    logger.info(f"Comparison saved to: {json_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing model bundles (default: models)'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['probability', 'price', 'weight'],
        help='Filter by task type'
    )
    
    args = parser.parse_args()
    
    compare_models(
        models_dir=args.models_dir,
        task_type=args.task
    )
