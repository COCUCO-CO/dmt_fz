"""
Gradient Boosting models for tabular data regression.

Supports XGBoost, LightGBM, and CatBoost for multi-output regression.
These are often state-of-the-art for tabular data, especially with small N.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try importing boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install with: pip install catboost")


class MultiOutputBoostingRegressor:
    """
    Wrapper for gradient boosting models to handle multi-output regression.
    Trains one model per target.
    """
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 n_targets: int = 23,
                 params: Dict[str, Any] = None):
        """
        Args:
            model_type: 'xgboost', 'lightgbm', or 'catboost'
            n_targets: Number of output targets
            params: Model-specific parameters
        """
        self.model_type = model_type.lower()
        self.n_targets = n_targets
        self.params = params or {}
        self.models = []
        
        # Validate availability
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        if self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        if self.model_type == 'catboost' and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")
    
    def _create_single_model(self):
        """Create a single model instance."""
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1
            }
            default_params.update(self.params)
            return xgb.XGBRegressor(**default_params)
        
        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1,
                'force_col_wise': True
            }
            default_params.update(self.params)
            return lgb.LGBMRegressor(**default_params)
        
        elif self.model_type == 'catboost':
            default_params = {
                'iterations': 100,
                'depth': 3,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'random_state': 42,
                'verbose': False,
                'thread_count': -1
            }
            default_params.update(self.params)
            return CatBoostRegressor(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit one model per target.
        
        Args:
            X: Features [n_samples, n_features]
            y: Targets [n_samples, n_targets]
        """
        self.models = []
        n_targets = y.shape[1] if y.ndim > 1 else 1
        
        for i in range(n_targets):
            model = self._create_single_model()
            y_i = y[:, i] if y.ndim > 1 else y
            model.fit(X, y_i)
            self.models.append(model)
        
        logger.info(f"Fitted {len(self.models)} {self.model_type} models")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict all targets.
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            Predictions [n_samples, n_targets]
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def get_feature_importance(self, feature_names: list = None) -> Dict[str, np.ndarray]:
        """Get feature importance from all models."""
        importances = []
        for model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        
        if importances:
            mean_importance = np.mean(importances, axis=0)
            std_importance = np.std(importances, axis=0)
            
            result = {
                'mean': mean_importance,
                'std': std_importance
            }
            
            if feature_names:
                # Sort by importance
                indices = np.argsort(mean_importance)[::-1]
                result['top_features'] = [(feature_names[i], mean_importance[i]) 
                                         for i in indices[:20]]
            
            return result
        return {}


def create_boosting_model(config: Dict[str, Any]) -> MultiOutputBoostingRegressor:
    """Factory function to create boosting model from config."""
    
    boosting_config = config.get('boosting', {})
    model_type = boosting_config.get('type', 'xgboost')
    
    params = {
        'n_estimators': boosting_config.get('n_estimators', 100),
        'max_depth': boosting_config.get('max_depth', 3),
        'learning_rate': boosting_config.get('learning_rate', 0.1),
        'subsample': boosting_config.get('subsample', 0.8),
        'colsample_bytree': boosting_config.get('colsample_bytree', 0.8),
        'reg_alpha': boosting_config.get('reg_alpha', 0.1),
        'reg_lambda': boosting_config.get('reg_lambda', 1.0),
    }
    
    return MultiOutputBoostingRegressor(
        model_type=model_type,
        n_targets=config['model']['output_dim'],
        params=params
    )


# Check what's available
def get_available_boosting_models():
    """Return list of available boosting models."""
    available = []
    if XGBOOST_AVAILABLE:
        available.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        available.append('lightgbm')
    if CATBOOST_AVAILABLE:
        available.append('catboost')
    return available


logger.info(f"Available boosting models: {get_available_boosting_models()}")

