#!/usr/bin/env python3
"""
AUTOMOTTO Predictive Maintenance Model Training
Trains ML models for vehicle health prediction
"""

import argparse
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from models.predictive_maintenance import PredictiveMaintenanceModel
from data.connector import DatabaseConnector
from data.feature_engine import FeatureEngineer
from utils.config import load_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_and_prepare_data(connector: DatabaseConnector, config: Dict) -> Tuple:
    """
    Load data from multiple vehicles and prepare for training
    """
    logger = logging.getLogger(__name__)
    
    all_features = []
    all_targets = []
    
    # Get all vehicles with sufficient history
    with connector.engine.connect() as conn:
        vehicles = conn.execute(
            "SELECT id FROM vehicles WHERE created_at <= NOW() - INTERVAL '90 days'"
        ).fetchall()
    
    logger.info(f"Found {len(vehicles)} vehicles for training")
    
    for vehicle in vehicles:
        try:
            vehicle_id = vehicle[0]
            
            # Load data for this vehicle
            telemetry = connector.get_vehicle_telemetry_data(vehicle_id, 365)
            maintenance = connector.get_maintenance_history(vehicle_id, 2)
            health = connector.get_health_snapshots(vehicle_id, 180)
            
            if len(telemetry) < config['data']['min_samples_per_vehicle']:
                logger.info(f"Skipping vehicle {vehicle_id}: insufficient data ({len(telemetry)} samples)")
                continue
            
            # Create features
            engineer = FeatureEngineer(config)
            features = engineer.create_features(telemetry, maintenance, health)
            
            # Prepare target (based on actual maintenance events)
            # For training, we simulate target based on maintenance patterns
            target = np.full(len(features), 0)  # Default: no immediate need
            
            # Mark samples close to actual maintenance as positive
            if not maintenance.empty:
                last_service = maintenance.iloc[0]
                service_odometer = last_service['odometer_km']
                days_since_service = (
                    datetime.now() - pd.to_datetime(last_service['performed_at'])
                ).days
                
                # Create window around service date
                service_window_start = service_odometer - 500  # 500km before
                service_window_end = service_odometer + 100   # 100km after
                
                # Mark samples within service window
                for idx, sample in enumerate(features.itertuples()):
                    sample_distance = sample.total_distance
                    if service_window_start <= sample_distance <= service_window_end:
                        if days_since_service > 30:
                            target[idx] = 1  # Routine
                        elif days_since_service > 7:
                            target[idx] = 2  # Warning
                        else:
                            target[idx] = 3  # Critical
    
            all_features.append(features)
            all_targets.append(pd.Series(target))
            
            logger.info(f"Added {len(features)} samples from vehicle {vehicle_id}")
            
        except Exception as e:
            logger.error(f"Error processing vehicle {vehicle.id}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No training data available")
    
    # Combine all vehicle data
    X = pd.concat(all_features, ignore_index=True)
    y = np.concatenate(all_targets)
    
    return X, y

def train_and_evaluate(connector: DatabaseConnector, config: Dict) -> Dict:
    """Complete training pipeline"""
    logger = logging.getLogger(__name__)
    
    # Load and prepare data
    X, y = load_and_prepare_data(connector, config)
    
    if len(X) == 0:
        logger.warning("No data available for training")
        return {}
    
    logger.info(f"Training with {len(X)} samples, {len(X.columns)} features")
    logger.info(f"Target distribution: {np.bincount(y)}")
    
    # Initialize model
    model = PredictiveMaintenanceModel(config)
    
    # Train the model
    results = model.train_classifier(X, y)
    
    logger.info(f"Model trained successfully")
    logger.info(f"Accuracy: {results['accuracy']:.3f}")
    logger.info(f"Best parameters: {results['best_params']}")
    
    # Display feature importance
    feature_importance = dict(sorted([
        (name, importance) for name, importance in results['feature_importance'].items()
    ], key=lambda x: x[1], reverse=True)[:10])
    
    logger.info("Top 10 important features:")
    for feature, importance in feature_importance.items():
        logger.info(f"  {feature}: {importance:.3f}")
    
    # Save model
    model_path = f"models/maintenance_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    if model.save_model(model_path):
        logger.info(f"Model saved to {model_path}")
    
    return {
        'status': 'success',
        'model_path': model_path,
        'accuracy': results['accuracy'],
        'feature_importance': feature_importance,
        'classification_report': results['classification_report'],
        'samples_used': len(X),
        'features_used': list(X.columns),
        'model_version': model.model_version
    }

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='AUTOMOTTO Predictive Maintenance Training')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--database-url', required=True, help='PostgreSQL connection string')
    parser.add_argument('--output-dir', default='models/', help='Output directory for model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize components
    connector = DatabaseConnector(args.database_url)
    logger = setup_logging()
    
    try:
        # Train model
        results = train_and_evaluate(connector, config)
        
        print("\n" + "="*60)
        print("AUTOMOTTO PREDICTIVE MAINTENANCE MODEL TRAINING RESULTS")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Model saved: {results['model_path']}")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Samples used: {results['samples_used']}")
        print(f"Features used: {len(results['features_used'])}")
        
        print("\nTop Features:")
        for feature, importance in results['feature_importance'].items():
            print(f"  {feature:25} {importance:6.3f}")
        
        print(f"\nModel Version: {results['model_version']}")
        print(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
