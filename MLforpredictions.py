import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import joblib
import pickle
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class PredictiveMaintenanceModel:
    """AI model for predicting vehicle maintenance needs"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.classifier = None
        self.regressor = None
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.model_version = "v1.0"
        self.is_fitted = False
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def train_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train classification model for maintenance urgency"""
        if len(X) < 50:
            raise ValueError("Insufficient training data. Need at least 50 samples.")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle categorical variables if any
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_train_numeric = X_train[numeric_features]
        X_test_numeric = X_test[numeric_features]
        
        # Train Random Forest Classifier
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        grid_search.fit(X_train_numeric, y_train)
        self.classifier = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test_numeric)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.feature_importance = dict(zip(
            numeric_features,
            self.classifier.feature_importances_
        ))
        
        self.is_fitted = True
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'best_params': grid_search.best_params_,
            'feature_importance': self.feature_importance
        }
    
    def predict_maintenance_urgency(self, X: pd.DataFrame) -> List[str]:
        """Predict maintenance urgency for new data"""
        if not self.is_fitted or self.classifier is None:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        # Ensure X has the same features as training data
        if set(X.columns) != set(self.classifier.feature_names_in_):
            raise ValueError("Feature mismatch. Expected columns: " + str(self.classifier.feature_names_in_))
        
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        category_map = {
            0: "No Action Needed",
            1: "Routine Maintenance",
            2: "Warning - Schedule Soon",
            3: "Critical - Immediate Attention"
        }
        
        return [
            category_map[pred] for pred in predictions
        ], probabilities
    
    def predict_days_until_failure(self, X: pd.DataFrame) -> np.ndarray:
        """Predict exact days until maintenance needed"""
        if self.regressor is None:
            # Train a separate regressor for days prediction
            self._train_regressor(X)
        
        return self.regressor.predict(X)
    
    def _train_regressor(self, X: pd.DataFrame, y_days: Optional[pd.Series] = None):
        """Train regression model for precise timing"""
        if y_days is None:
            # If no regression target, create synthetic one
            y_days = np.random.randint(7, 365, size=len(X))
        
        from sklearn.ensemble import RandomForestRegressor
        
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        rf_reg.fit(X, y_days)
        self.regressor = rf_reg
    
    def get_risk_factors(self, X: pd.DataFrame) -> List[Dict]:
        """Generate human-readable risk factors"""
        if self.feature_importance is None:
            return []
        
        risk_factors = []
        for i, (feature, importance) in enumerate(self.feature_importance.items()):
            if importance > 0.05:  # Only significant features
                weight = importance * 100
                risk_desc = self._get_feature_risk_description(feature, X[feature].mean())
                if risk_desc:
                    risk_factors.append({
                        'feature': feature,
                        'importance': weight,
                        'risk_level': 'high' if weight > 15 else 'medium',
                        'description': risk_desc,
                        'value': round(X[feature].mean(), 2)
                    })
        
        return sorted(risk_factors, key=lambda x: x['importance'], reverse=True)
    
    def _get_feature_risk_description(self, feature: str, value: float) -> Optional[str]:
        """Get human-readable risk description for feature"""
        risk_descriptions = {
            'total_distance': 'High mileage increases wear on brakes, tires, and suspension',
            'avg_speed': 'Higher average speeds accelerate engine and transmission wear',
            'harsh_driving_score': 'Aggressive driving increases component stress and failure rate',
            'avg_engine_rpm': 'Higher RPM operation accelerates engine wear and oil degradation',
            'throttle_usage_avg': 'Frequent heavy throttle use indicates aggressive driving patterns',
            'avg_coolant_temp': 'Elevated temperatures can indicate cooling system issues',
            'battery_degradation_rate': 'Rapid battery degradation suggests electrical system problems',
            'days_since_oil_change': 'Oil change overdue by more than 30 days',
            'current_oil_life_pct': 'Oil life below 20% indicates immediate maintenance needed',
            'dtc_count_recent': 'Multiple diagnostic trouble codes suggest underlying issues'
        }
        
        return risk_descriptions.get(feature)
    
    def generate_maintenance_recommendations(self, predictions: List[str], 
                                          risk_factors: List[Dict]) -> List[Dict]:
        """Generate actionable maintenance recommendations"""
        recommendations = []
        
        for i, pred in enumerate(predictions):
            if pred == "Critical - Immediate Attention":
                recommendations.append({
                    'priority': 'critical',
                    'action': 'Schedule maintenance appointment immediately',
                    'urgency': 'HIGH - Stop driving if possible',
                    'estimated_cost': '$300-$1200',
                    'risk_factors': [f['description'] for f in risk_factors[:3]]
                })
            elif pred == "Warning - Schedule Soon":
                recommendations.append({
                    'priority': 'warning',
                    'action': 'Schedule service within next 7 days',
                    'urgency': 'MEDIUM - Do not delay',
                    'estimated_cost': '$150-$600',
                    'risk_factors': [f['description'] for f in risk_factors[:2]]
                })
            elif pred == "Routine Maintenance":
                recommendations.append({
                    'priority': 'routine',
                    'action': 'Schedule regular maintenance soon',
                    'urgency': 'LOW',
                    'estimated_cost': '$50-$200',
                    'risk_factors': [f['description'] for f in risk_factors[:1]]
                })
            else:
                recommendations.append({
                    'priority': 'healthy',
                    'action': 'Continue normal maintenance schedule',
                    'urgency': 'NORMAL',
                    'estimated_cost': '$0',
                    'risk_factors': []
                })
        
        return recommendations
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_names': list(self.classifier.feature_names_in_) if self.classifier else [],
            'model_version': self.model_version,
            'feature_importance': self.feature_importance,
            'training_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'wb') as f:
                joblib.dump(model_data, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = joblib.load(f)
            
            self.classifier = model_data['classifier']
            self.regressor = model_data['regressor']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            self.model_version = model_data['model_version']
            self.is_fitted = True
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

class AnomalyDetector:
    """Anomaly detection for unusual vehicle behavior"""
    
    def __init__(self):
        self.isolation_forest = None
        self.threshold = 0.02  # Anomaly threshold
    
    def fit(self, X: pd.DataFrame):
        """Train anomaly detection model"""
        self.isolation_forest = IsolationForest(
            contamination=0.05,  # 5% of data expected to be anomalies
            random_state=42
        )
        self.isolation_forest.fit(X)
    
    def detect_anomalies(self, X: pd.DataFrame) -> pd.Series:
        """Detect anomalous readings"""
        if self.isolation_forest is None:
            raise ValueError("Anomaly detector not fitted")
        
        anomaly_scores = self.isolation_forest.decision_function(X)
        is_anomaly = anomaly_scores < self.threshold
        
        return is_anomaly
