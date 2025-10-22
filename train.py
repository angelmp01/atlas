"""
Script de entrenamiento para los modelos de ATLAS ML.

Este script entrena los 3 modelos principales:
1. Probability (Regimen B)
2. Price
3. Weight

Uso:
    python train.py                    # Entrena todos los modelos
    python train.py --only probability # Solo probability
    python train.py --only price       # Solo price
    python train.py --only weight      # Solo weight
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

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


def train_all_models(config: Config):
    """
    Entrena todos los modelos del sistema.
    
    Args:
        config: Configuracion para el entrenamiento
        
    Returns:
        dict: Resultados del entrenamiento con status de cada modelo
    """
    logger.info("="*60)
    logger.info("INICIO DEL ENTRENAMIENTO DE MODELOS ATLAS ML")
    logger.info("="*60)
    
    results = {}
    
    # 1. Entrenar modelo de probabilidad (Regimen B)
    logger.info("\n[1/3] Entrenando modelo de PROBABILIDAD (Regimen B)...")
    try:
        prob_model = train_probability(config)
        results['probability'] = {
            'status': 'success',
            'bundle': prob_model,
            'path': prob_model.model_path
        }
        logger.info(f"[OK] Modelo de probabilidad guardado en: {prob_model.model_path}")
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo de probabilidad: {e}")
        results['probability'] = {'status': 'failed', 'error': str(e)}
    
    # 2. Entrenar modelo de precio
    logger.info("\n[2/3] Entrenando modelo de PRECIO...")
    try:
        price_model = train_price(config)
        results['price'] = {
            'status': 'success',
            'bundle': price_model,
            'path': price_model.model_path
        }
        logger.info(f"[OK] Modelo de precio guardado en: {price_model.model_path}")
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo de precio: {e}")
        results['price'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Entrenar modelo de peso
    logger.info("\n[3/3] Entrenando modelo de PESO...")
    try:
        weight_model = train_weight(config)
        results['weight'] = {
            'status': 'success',
            'bundle': weight_model,
            'path': weight_model.model_path
        }
        logger.info(f"[OK] Modelo de peso guardado en: {weight_model.model_path}")
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


def train_single_model(model_type: str, config: Config):
    """
    Entrena un solo modelo.
    
    Args:
        model_type: 'probability', 'price' o 'weight'
        config: Configuracion para el entrenamiento
        
    Returns:
        Model bundle entrenado
    """
    logger.info(f"Entrenando modelo: {model_type.upper()}")
    
    try:
        if model_type == 'probability':
            model = train_probability(config)
        elif model_type == 'price':
            model = train_price(config)
        elif model_type == 'weight':
            model = train_weight(config)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
        
        logger.info(f"[OK] Modelo guardado en: {model.model_path}")
        return model
        
    except Exception as e:
        logger.error(f"[ERROR] Error entrenando modelo {model_type}: {e}")
        raise


def main():
    """Funcion principal del script."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelos de ATLAS ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python train.py                         # Entrena todos los modelos
  python train.py --only probability      # Solo probability
  python train.py --only price            # Solo price
  python train.py --only weight           # Solo weight
  python train.py --regime B              # Especificar regimen (A o B)
        """
    )
    
    parser.add_argument(
        '--only',
        type=str,
        choices=['probability', 'price', 'weight'],
        help='Entrenar solo un modelo especifico'
    )
    
    parser.add_argument(
        '--regime',
        type=str,
        choices=['A', 'B'],
        default='B',
        help='Regimen de entrenamiento para probability (default: B)'
    )
    
    args = parser.parse_args()
    
    # Crear directorio de logs si no existe
    Path('logs').mkdir(exist_ok=True)
    
    # Cargar configuracion
    logger.info("Cargando configuracion...")
    config = Config()
    
    # Configurar regimen si es probability
    config.training.training_regime = f'regime_{args.regime.lower()}'
    logger.info(f"Regimen de entrenamiento: {config.training.training_regime}")
    
    # Entrenar
    try:
        if args.only:
            # Entrenar un solo modelo
            train_single_model(args.only, config)
            logger.info("\n[SUCCESS] Entrenamiento completado exitosamente!")
            return 0
        else:
            # Entrenar todos los modelos
            results = train_all_models(config)
            
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


if __name__ == '__main__':
    sys.exit(main())
