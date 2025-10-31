import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime, timedelta

class FeatureEngineer:
    """Feature engineering for predictive maintenance model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def create_features(self, telemetry_df: pd.DataFrame, 
                       maintenance_df: pd.DataFrame, 
                       health_df: pd.DataFrame) -> pd.DataFrame:
        """Create predictive features from raw data"""
        if telemetry_df.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame()
        
        # Basic vehicle usage features
        features['total_distance'] = telemetry_df['distance_km'].max() or 0
        features['avg_daily_distance'] = telemetry_df['distance_km'].sum() / 365
        
        # Speed and driving style
        features['avg_speed'] = telemetry_df['avg_speed_kph'].mean()
        features['max_speed'] = telemetry_df['avg_speed_kph'].max()
        features['speed_variance'] = telemetry_df['avg_speed_kph'].std()
        features['harsh_driving_score'] = (
            telemetry_df['harsh_brake_flag'].sum() * 0.6 +
            telemetry_df['rapid_accel_flag'].sum() * 0.4
        ) / len(telemetry_df)
        
        # Engine and fuel system
        features['avg_engine_rpm'] = telemetry_df['engine_rpm'].mean()
        features['engine_rpm_std'] = telemetry_df['engine_rpm'].std()
        features['throttle_usage_avg'] = telemetry_df['throttle_pct'].mean()
        features['throttle_usage_max'] = telemetry_df['throttle_pct'].max()
        features['fuel_efficiency_trend'] = telemetry_df['efficiency_penalty'].rolling(
            window=100, min_periods=10
        ).mean().iloc[-1] or 0
        
        # Temperature and thermal management
        features['avg_coolant_temp'] = telemetry_df['coolant_temp_c'].mean()
        features['max_coolant_temp'] = telemetry_df['coolant_temp_c'].max()
        features['temp_anomaly_count'] = (telemetry_df['temp_anomaly'] > 5).sum()
        features['cooling_system_stress'] = (
            (telemetry_df['coolant_temp_c'] > 100).sum() * 0.8 +
            (telemetry_df['temp_anomaly'] > 10).sum() * 0.2
        )
        
        # Battery and electrical
        features['avg_battery_soc'] = telemetry_df['battery_soc_pct'].mean()
        features['battery_degradation_rate'] = telemetry_df['battery_degradation'].rolling(
            window=50, min_periods=10
        ).mean().iloc[-1] or 0
        features['electrical_load_avg'] = telemetry_df['engine_load_pct'].mean()
        
        # Maintenance history features
        if not maintenance_df.empty:
            recent_maintenance = maintenance_df[maintenance_df['type'].isin([
                'oil_change', 'brake_service', 'tire_rotation'
            ])].sort_values('performed_at', ascending=False)
            
            if len(recent_maintenance) > 0:
                days_since_last_service = (
                    datetime.now() - pd.to_datetime(recent_maintenance['performed_at'].iloc[0])
                ).days
                
                features['days_since_oil_change'] = days_since_last_service if recent_maintenance['type'].iloc[0] == 'oil_change' else 365
                features['miles_since_oil_change'] = (
                    features['total_distance'] - recent_maintenance['odometer_km'].iloc[0]
                ) if not recent_maintenance.empty else 10000
                
                # Calculate service intervals
                service_intervals = {
                    'oil_change': 5000,  # km
                    'brake_service': 40000,
                    'tire_rotation': 10000,
                    'transmission': 80000
                }
                
                for service_type, interval_km in service_intervals.items():
                    last_service = maintenance_df[
                        maintenance_df['type'] == service_type
                    ].sort_values('performed_at', ascending=False)
                    
                    if len(last_service) > 0:
                        miles_since = features['total_distance'] - last_service['odometer_km'].iloc[0]
                        features[f'miles_since_{service_type.replace("_", "-")}'] = miles_since
                    else:
                        features[f'miles_since_{service_type.replace("_", "-")}'] = interval_km
        
        # Health snapshot features
        if not health_df.empty:
            recent_health = health_df.sort_values('ts', ascending=False).iloc[0]
            features['current_oil_life_pct'] = recent_health['oil_life_pct'] or 80
            features['battery_health_pct'] = recent_health['battery_health_pct'] or 90
            features['dtc_count_recent'] = len(recent_health['dtc_codes']) if recent_health['dtc_codes'] else 0
        
        # Temporal features
        features['total_trips'] = len(telemetry_df)
        features['active_days'] = len(telemetry_df['ts'].dt.date.unique())
        features['weekend_usage_ratio'] = (
            telemetry_df['ts'].dt.weekday >= 5
        ).mean()
        
        # Environmental factors
        features['seasonal_factor'] = 1.2 if datetime.now().month in [1, 2, 12] else 1.0  # Winter penalty
        features['driving_style_risk'] = 1 + (1 - (features['avg_driver_score'] or 80) / 100) * 0.5
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select top predictive features using statistical methods"""
        if self.feature_selector is None:
            # Define feature importance based on domain knowledge
            important_features = [
                'total_distance',
                'avg_speed',
                'max_speed',
                'harsh_driving_score',
                'avg_engine_rpm',
                'throttle_usage_avg',
                'avg_coolant_temp',
                'battery_degradation_rate',
                'days_since_oil_change',
                'current_oil_life_pct',
                'dtc_count_recent'
            ]
            
            return X[important_features].fillna(0)
        
        return self.feature_selector.transform(X)
    
    def scale_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Scale features for ML model input"""
        # Remove non-numeric columns first
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_scaled = X[numeric_columns]
        
        # Fit scaler if not fitted
        if not hasattr(self.scaler, 'scale_') or len(self.scaler.scale_) == 0:
            X_scaled = self.scaler.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler.transform(X_scaled)
        
        return pd.DataFrame(X_scaled, columns=numeric_columns), self.scaler
    
    def prepare_training_data(self, vehicle_id: str, target_days: int = 90) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare supervised training data for a vehicle"""
        # Get historical data
        telemetry = self.connector.get_vehicle_telemetry_data(vehicle_id, 365)
        maintenance = self.connector.get_maintenance_history(vehicle_id, 2)
        health = self.connector.get_health_snapshots(vehicle_id, 180)
        
        if telemetry.empty:
            return pd.DataFrame(), pd.Series()
        
        # Create features
        features = self.create_features(telemetry, maintenance, health)
        
        # Create target variable (days until next service)
        # This is a simplified example - in production, you'd use actual failure data
        target = []
        for _, row in features.iterrows():
            days_since_last_service = row.get('days_since_oil_change', 365)
            
            if days_since_last_service <= 7:
                target.append(3)  # Critical
            elif days_since_last_service <= 30:
                target.append(2)  # Warning
            elif days_since_last_service <= 90:
                target.append(1)  # Routine
            else:
                target.append(0)  # No action needed
        
        features['target_days'] = target
        
        X = self.select_features(features)
        y = pd.Series(target)
        
        return X, y
