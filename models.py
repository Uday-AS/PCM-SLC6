from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def find_best_params_rf(X_train, y_train, param_grid=None, random_state=42):
    """
    Find best parameters for Random Forest using GridSearchCV
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target values
    n_estimators : int, default=100
        Maximum number of trees to try
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict : Best parameters found through grid search
    """

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 400, 800, 1500, 2000],
            'max_depth': [10, 30, 60, 80, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    
    rf_model = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=10,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_

def train_random_forest(X_train, y_train, best_params, random_state=42):
    """
    Train Random Forest model with best parameters on full training data
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target values  
    best_params : dict
        Best parameters found through grid search
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    rf_model : RandomForestRegressor
        Trained Random Forest model
    """
    rf_model = RandomForestRegressor(**best_params, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def find_best_params_xgb(X_train, y_train, param_grid=None, random_state=42):
    """
    Find best parameters for XGBoost using GridSearchCV
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target values
    n_estimators : int, default=100
        Maximum number of boosting rounds to try
    learning_rate : float, default=0.1
        Default learning rate to try
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict : Best parameters found through grid search
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'learning_rate': [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41],
            'max_depth': [5, 10, 15, 20, 25, 30],
        }
    
    xgb_model = XGBRegressor(random_state=random_state)
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=10,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_

def train_xgboost(X_train, y_train, best_params, random_state=42):
    """
    Train XGBoost model with best parameters on full training data
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target values
    best_params : dict
        Best parameters found through grid search
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    xgb_model : XGBRegressor
        Trained XGBoost model
    """
    xgb_model = XGBRegressor(**best_params, random_state=random_state)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def predict_and_evaluate(model, X_test, y_test):
    """
    Make predictions and evaluate model performance
    
    Parameters:
    -----------
    model : estimator object
        Trained model (Random Forest or XGBoost)
    X_test : array-like
        Test features
    y_test : array-like
        True test values
        
    Returns:
    --------
    dict : Dictionary containing various performance metrics
    """
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False)
    }
    
    return metrics
