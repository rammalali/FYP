import pandas as pd
import numpy as np
import os, pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from uuid import uuid4
from sklearn.metrics import r2_score

class Train:
    def __init__(self, df, model_type):
        self.df = df
        self.type = model_type
        self.check_df_data()

    def check_df_data(self):
        if all(col in self.df.columns for col in ['velocity', 'Cycle_Number']):
            if self.type == "Elastic Displacement":
                return 'smoothed_displacement' in (col.lower() for col in self.df.columns)
            elif self.type == "Permanent Settlement":
                return 'smoothed_settlement' in (col.lower() for col in self.df.columns)
            elif self.type == "Acceleration":
                return all(
                    col.lower() in (c.lower() for c in self.df.columns)
                    for col in ['smoothed_positive_max', 'smoothed_negative_max']
                )
        return False
    

    def save_model(self, model, type, path):
        """Save the trained model to the specified path."""
        uuid = uuid4()
        model_name = f"{type.replace(' ', '_').lower()}_{uuid}_model.pkl"
        full_path = os.path.join(path, model_name)
        with open(full_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        return full_path
    
    def train_model(self):
        if self.type == "Elastic Displacement":
            return self._train_displacement_model()
        elif self.type == "Permanent Settlement":
            return self._train_settlement_model()
        elif self.type == "Acceleration":
            return (self._train_neg_acceleration_model(), self._train_pos_acceleration_model())
        

    def _train_displacement_model(self):
        X = self.df[['velocity', 'Cycle_Number']]
        y = self.df['smoothed_displacement']


        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


        def custom_scorer(y_true, y_pred):
            score = 1 - mean_absolute_error(y_true, y_pred)
            return score

        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 16],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 4, 8],
            'max_features': ['auto', 'sqrt']
        }


        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring=make_scorer(custom_scorer),
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)
        smoothed_displacement_rf_model = grid_search.best_estimator_

        smoothed_displacement_rf_model.fit(X_train_val, y_train_val) 

        y_pred_test = smoothed_displacement_rf_model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)

        return smoothed_displacement_rf_model, r2_test 

    def _train_settlement_model(self):
        X = self.df[['velocity', 'Cycle_Number']]
        y = self.df['smoothed_settlement']

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


        def custom_scorer(y_true, y_pred):
            score = 1 - mean_absolute_error(y_true, y_pred)
            return score

        # Define the model and hyperparameters to tune
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 12],
            'min_samples_split': [20, 24],
            'min_samples_leaf': [8, 16],
            'max_features': ['sqrt']
        }


        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring=make_scorer(custom_scorer),
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)
        ps_rf_model = grid_search.best_estimator_

        ps_rf_model.fit(X_train_val, y_train_val)

        y_pred_test = ps_rf_model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)

        return ps_rf_model, r2_test
    
    def _train_neg_acceleration_model(self):
        X = self.df[['velocity', 'Cycle_Number']]
        y_neg = self.df['smoothed_negative_max']

        # Splitting data into train/validation/test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_neg, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
        # This results in 60% train, 20% validation, and 20% test splits

        # Custom scoring function
        def custom_scorer(y_true, y_pred):
            # Using the inverse of MAE as the scoring metric
            score = 1 - mean_absolute_error(y_true, y_pred)
            return score

        # Define the model and hyperparameters to tune
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [150],
            'max_depth': [16, 18],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1],
            'max_features': ['auto', 'sqrt']
        }

        # Set up the grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring=make_scorer(custom_scorer),
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)
        n_max_rf_model = grid_search.best_estimator_

        # Evaluate on the test set
        n_max_rf_model.fit(X_train_val, y_train_val)
        y_pred_test = n_max_rf_model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        return n_max_rf_model, r2_test
    
    def _train_pos_acceleration_model(self):
        X = self.df[['velocity', 'Cycle_Number']]
        y = self.df['smoothed_positive_max']

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


        def custom_scorer(y_true, y_pred):
            score = 1 - mean_absolute_error(y_true, y_pred)
            return score

        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [150],
            'max_depth': [16, 18],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1],
            'max_features': ['auto', 'sqrt']
        }

        # Set up the grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring=make_scorer(custom_scorer),
            n_jobs=-1,
            verbose=2
        )

        # Train the model with the best parameters using cross-validation on the train set
        grid_search.fit(X_train, y_train)
        p_max_rf_model = grid_search.best_estimator_

        # Evaluate on the test set
        p_max_rf_model.fit(X_train_val, y_train_val)  # Train on full train/validation data
        y_pred_test = p_max_rf_model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)

        return p_max_rf_model, r2_test


    def _process_row_data(self):
        self.df['smoothed_displacement'] = np.nan
        
        for index in self.df.index:
            data_to_smooth = pd.to_numeric(self.df.iloc[index, 19:139], errors='coerce')
            
            smoothed_data = data_to_smooth.rolling(window=5, center=True).mean()
            
            first_non_nan_value = smoothed_data[smoothed_data.notna()].iloc[0] if smoothed_data.notna().any() else np.nan
            
            self.df.at[index, 'smoothed_displacement'] = first_non_nan_value
                
