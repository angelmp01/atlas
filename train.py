"""
Script de entrenamiento para los modelos de ATLAS ML.

Este script entrena los 3 modelos principales:
1. Probability (daily trip counts)
2. Price
3. Weight

Uso:
    python train.py                    # Entrena todos los modelos
    python train.py --only probability # Solo probability
    python train.py --only price       # Solo price
    python train.py --only weight      # Solo weight
    python train.py --no-wandb         # Sin logging a wandb
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

# Importar funciones de entrenamiento
from atlas_ml import train_probability, train_price, train_weight, Config
from atlas_ml.wandb_logger import WandbLogger


def train_all_models(config: Config, wandb_logger: Optional[WandbLogger] = None, quick_test: bool = False):
    """
    Entrena todos los modelos del sistema.
    
    Args:
        config: Configuracion para el entrenamiento
        wandb_logger: Optional wandb logger for experiment tracking
        quick_test: If True, use only a few OD pairs for fast testing
        
    Returns:
        dict: Resultados del entrenamiento con status de cada modelo
    """
    logger.info("="*60)
    logger.info("INICIO DEL ENTRENAMIENTO DE MODELOS ATLAS ML")
    if quick_test:
        logger.info("*** QUICK TEST MODE: Training on 5 OD pairs only ***")
    logger.info("="*60)
    
    results = {}
    
    # 1. Entrenar modelo de probabilidad
    logger.info("\n[1/3] Entrenando modelo de PROBABILIDAD...")
    logger.info("="*19)
    try:
        prob_model = train_probability(config, wandb_logger=wandb_logger, quick_test=quick_test)
        results['probability'] = {
            'status': 'success',
            'bundle': prob_model,
            'path': prob_model.model_path
        }
        logger.info(f"[OK] Modelo de probabilidad guardado en: {prob_model.model_path}")
        
        # Log model artifact to wandb
        if wandb_logger and wandb_logger.enabled:
            import json
            model_card_path = prob_model.model_path / "model_card.json"
            if model_card_path.exists():
                with open(model_card_path) as f:
                    model_card = json.load(f)
                    wandb_logger.log_model_bundle(
                        model_path=prob_model.model_path,
                        task="probability",
                        metrics=model_card.get("performance", {})
                    )
        
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo de probabilidad: {e}")
        results['probability'] = {'status': 'failed', 'error': str(e)}
    
    # 2. Entrenar modelo de precio
    logger.info("\n[2/3] Entrenando modelo de PRECIO...")
    try:
        price_model = train_price(config, wandb_logger=wandb_logger, quick_test=quick_test)
        results['price'] = {
            'status': 'success',
            'bundle': price_model,
            'path': price_model.model_path
        }
        logger.info(f"[OK] Modelo de precio guardado en: {price_model.model_path}")
        
        # Log model artifact to wandb
        if wandb_logger and wandb_logger.enabled:
            import json
            model_card_path = price_model.model_path / "model_card.json"
            if model_card_path.exists():
                with open(model_card_path) as f:
                    model_card = json.load(f)
                    wandb_logger.log_model_bundle(
                        model_path=price_model.model_path,
                        task="price",
                        metrics=model_card.get("performance", {})
                    )
        
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo de precio: {e}")
        results['price'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Entrenar modelo de peso
    logger.info("\n[3/3] Entrenando modelo de PESO...")
    try:
        weight_model = train_weight(config, wandb_logger=wandb_logger, quick_test=quick_test)
        results['weight'] = {
            'status': 'success',
            'bundle': weight_model,
            'path': weight_model.model_path
        }
        logger.info(f"[OK] Modelo de peso guardado en: {weight_model.model_path}")
        
        # Log model artifact to wandb
        if wandb_logger and wandb_logger.enabled:
            import json
            model_card_path = weight_model.model_path / "model_card.json"
            if model_card_path.exists():
                with open(model_card_path) as f:
                    model_card = json.load(f)
                    wandb_logger.log_model_bundle(
                        model_path=weight_model.model_path,
                        task="weight",
                        metrics=model_card.get("performance", {})
                    )
        
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo de peso: {e}")
        results['weight'] = {'status': 'failed', 'error': str(e)}
    
    # Resumen
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DEL ENTRENAMIENTO")
    logger.info("="*60)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    for model_name, result in results.items():
        status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
        logger.info(f"{status_icon} {model_name.upper()}: {result['status']}")
        if result['status'] == 'success':
            logger.info(f"   -> Guardado en: {result['path']}")
        else:
            logger.info(f"   -> Error: {result.get('error', 'Unknown')}")
    
    logger.info(f"\nModelos entrenados exitosamente: {success_count}/{total_count}")
    logger.info("="*60)
    
    return results


def train_single_model(model_type: str, config: Config, wandb_logger=None, quick_test: bool = False):
    """
    Entrena un solo modelo.
    
    Args:
        model_type: 'probability', 'price' o 'weight'
        config: Configuracion para el entrenamiento
        wandb_logger: Optional WandbLogger for logging metrics
        quick_test: If True, use only a few OD pairs for fast testing
        
    Returns:
        Model bundle entrenado
    """
    logger.info(f"Entrenando modelo: {model_type.upper()}")
    
    try:
        if model_type == 'probability':
            model = train_probability(config, wandb_logger=wandb_logger, quick_test=quick_test)
        elif model_type == 'price':
            model = train_price(config, wandb_logger=wandb_logger, quick_test=quick_test)
        elif model_type == 'weight':
            model = train_weight(config, wandb_logger=wandb_logger, quick_test=quick_test)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
        
        logger.info(f"[OK] Modelo guardado en: {model.model_path}")
        return model
        
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo {model_type}: {e}")
        raise


def main():
    """Funcion principal del script."""
    # Suppress pydantic warnings from wandb
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
    
    parser = argparse.ArgumentParser(
        description='Entrenar modelos de ATLAS ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python train.py                         # Entrena todos los modelos
  python train.py --only probability      # Solo probability
  python train.py --only price            # Solo price
  python train.py --only weight           # Solo weight
  python train.py --no-wandb              # Sin logging a wandb
  python train.py --quick-test            # Quick test (5 OD pairs, ~1-2 min)
  python train.py --quick-test --only probability  # Quick test solo probability
        """
    )
    
    parser.add_argument(
        '--only',
        type=str,
        choices=['probability', 'price', 'weight'],
        help='Entrenar solo un modelo especifico'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Deshabilitar logging a wandb'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: train only on a few OD pairs (fast, for testing wandb/code)'
    )
    
    args = parser.parse_args()
    
    # Crear directorio de logs si no existe
    Path('logs').mkdir(exist_ok=True)
    
    # Cargar configuracion
    logger.info("Cargando configuracion...")
    config = Config()
    
    # Initialize wandb logger
    wandb_logger = None
    if not args.no_wandb:
        try:
            # Get GPU info for tags
            import platform
            tags = []
            
            # Add quick-test tag if enabled
            if args.quick_test:
                tags.append('quick-test')
            
            # Get XGBoost params
            xgb_params = config.model.xgb_params
            
            # Check if GPU is being used
            if xgb_params.get('tree_method') == 'hist' and xgb_params.get('device') == 'cuda':
                tags.append('gpu')
            else:
                tags.append('cpu')
            
            # Add platform tag
            tags.append(platform.system().lower())
            
            wandb_logger = WandbLogger(
                project="atlas-ml",
                entity="pgaitl",
                name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=tags,
                config={
                    "model": {
                        "tree_method": xgb_params.get('tree_method', 'hist'),
                        "device": xgb_params.get('device', 'cpu'),
                        "n_estimators": xgb_params.get('n_estimators', 500),
                        "max_depth": xgb_params.get('max_depth', 10),
                        "learning_rate": xgb_params.get('learning_rate', 0.05),
                        "subsample": xgb_params.get('subsample', 0.8),
                        "colsample_bytree": xgb_params.get('colsample_bytree', 0.8)
                    },
                    "cv": {
                        "months_train": config.training.cv_months_train,
                        "months_test": config.training.cv_months_test
                    }
                },
                enabled=True
            )
            logger.info("wandb logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
            wandb_logger = None
    else:
        logger.info("wandb logging disabled (--no-wandb flag)")
    
    # Entrenar
    try:
        if args.only:
            # Entrenar un solo modelo
            train_single_model(args.only, config, wandb_logger=wandb_logger, quick_test=args.quick_test)
            logger.info("\n[SUCCESS] Entrenamiento completado exitosamente!")
            return 0
        else:
            # Entrenar todos los modelos
            results = train_all_models(config, wandb_logger=wandb_logger, quick_test=args.quick_test)
            
            # Verificar si al menos un modelo se entrenó correctamente
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            
            if success_count == 0:
                logger.error("\n[FAILURE] Ningun modelo se entrenó correctamente.")
                return 1
            elif success_count < len(results):
                logger.warning(f"\n[PARTIAL] Entrenamiento parcialmente exitoso: {success_count}/{len(results)} modelos.")
                return 0  # Exit code 0 porque al menos algunos modelos funcionaron
            else:
                logger.info("\n[SUCCESS] Todos los modelos entrenados exitosamente!")
                return 0
        
    except Exception as e:
        logger.error(f"\n[FAILURE] Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Finish wandb run
        if wandb_logger:
            wandb_logger.finish()


if __name__ == '__main__':
    sys.exit(main())
