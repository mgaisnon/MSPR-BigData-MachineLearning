from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import joblib
import logging
from config.config import MODEL_CONFIG
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ElectionPredictor:
    """Classe principale pour la prédiction électorale"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.feature_importance = None
        self.cv_results = {}
    
    def initialize_models(self):
        """Initialise les différents modèles à tester"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=MODEL_CONFIG.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG.random_state,
                max_depth=10,
                min_samples_split=5
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG.random_state,
                max_depth=6,
                learning_rate=0.1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG.random_state,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG.random_state,
                max_depth=6,
                learning_rate=0.1,
                verbose=-1
            ),
            'svm': SVC(
                kernel='rbf',
                random_state=MODEL_CONFIG.random_state,
                probability=True,
                C=1.0
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }
        
        logger.info(f"Initialisé {len(self.models)} modèles")
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = None) -> Dict[str, Dict]:
        """Entraîne tous les modèles et retourne leurs performances"""
        if test_size is None:
            test_size = MODEL_CONFIG.test_size
        
        results = {}
        
        # Vérification des données
        if X.empty or y.empty:
            logger.error("Données d'entrée vides")
            return results
        
        # Split train/test stratifié
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=MODEL_CONFIG.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
        except ValueError as e:
            logger.warning(f"Stratification impossible: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=MODEL_CONFIG.random_state
            )
        
        logger.info(f"Entraînement sur {len(X_train)} échantillons, test sur {len(X_test)}")
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=MODEL_CONFIG.cv_folds, shuffle=True, random_state=MODEL_CONFIG.random_state)
        
        for name, model in self.models.items():
            logger.info(f"Entraînement du modèle: {name}")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Évaluation
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as cv_error:
                    logger.warning(f"CV failed for {name}: {cv_error}")
                    cv_mean = accuracy
                    cv_std = 0
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # AUC si binaire et probabilités disponibles
                auc_score = None
                if len(np.unique(y)) == 2 and y_pred_proba is not None:
                    try:
                        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                    except Exception:
                        pass
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'auc_score': auc_score
                }
                
                # Mise à jour du meilleur modèle
                if cv_mean > self.best_score:
                    self.best_score = cv_mean
                    self.best_model = model
                    self.best_model_name = name
                
                logger.info(f"{name} - Accuracy: {accuracy:.3f}, CV: {cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.cv_results = results
        logger.info(f"Meilleur modèle: {self.best_model_name} (CV Score: {self.best_score:.3f})")
        
        return results
    
    def optimize_best_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimise les hyperparamètres du meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        # Grilles de paramètres selon le type de modèle
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8],
                'learning_rate': [0.1, 0.2],
                'num_leaves': [31, 50]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8],
                'learning_rate': [0.1, 0.2]
            }
        }
        
        if self.best_model_name not in param_grids:
            logger.warning(f"Pas d'optimisation définie pour {self.best_model_name}")
            return {'message': 'Pas d\'optimisation disponible pour ce modèle'}
        
        logger.info(f"Optimisation des hyperparamètres pour {self.best_model_name}")
        
        try:
            # Cross-validation stratifiée
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_CONFIG.random_state)
            
            grid_search = GridSearchCV(
                estimator=type(self.best_model)(**self.best_model.get_params()),
                param_grid=param_grids[self.best_model_name],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            # Mise à jour du meilleur modèle
            self.best_model = grid_search.best_estimator_
            old_score = self.best_score
            self.best_score = grid_search.best_score_
            
            logger.info(f"Score avant optimisation: {old_score:.3f}")
            logger.info(f"Score après optimisation: {self.best_score:.3f}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'score_improvement': self.best_score - old_score,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Retourne l'importance des features"""
        if self.best_model is None:
            return pd.DataFrame()
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                importance = np.abs(self.best_model.coef_[0])
            else:
                logger.warning("Pas d'importance des features disponible")
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Normaliser les importances
            importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
            
            self.feature_importance = importance_df
            return importance_df
            
        except Exception as e:
            logger.error(f"Erreur calcul importance: {e}")
            return pd.DataFrame()
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fait des prédictions avec le meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        try:
            predictions = self.best_model.predict(X)
            probabilities = None
            
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Évalue le modèle sur des données de test"""
        if self.best_model is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        try:
            predictions, probabilities = self.predict(X_test)
            
            evaluation = {
                'accuracy': accuracy_score(y_test, predictions),
                'classification_report': classification_report(y_test, predictions, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, predictions)
            }
            
            # AUC si binaire
            if len(np.unique(y_test)) == 2 and probabilities is not None:
                evaluation['auc_score'] = roc_auc_score(y_test, probabilities[:, 1])
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Erreur évaluation: {e}")
            return {}
    
    def save_model(self, filename: str = None):
        """Sauvegarde le meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        if filename is None:
            filename = f"best_model_{self.best_model_name}.pkl"
        
        filepath = MODEL_CONFIG.models_dir / filename
        
        try:
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'score': self.best_score,
                'feature_importance': self.feature_importance,
                'cv_results': self.cv_results
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Modèle sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            raise
    
    def load_model(self, filename: str = None):
        """Charge un modèle sauvegardé"""
        if filename is None:
            filename = "best_election_model.pkl"
        
        filepath = MODEL_CONFIG.models_dir / filename
        
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.best_score = model_data['score']
            self.feature_importance = model_data.get('feature_importance')
            self.cv_results = model_data.get('cv_results', {})
            
            logger.info(f"Modèle chargé: {filepath}")
            logger.info(f"Modèle: {self.best_model_name}, Score: {self.best_score:.3f}")
            
        except Exception as e:
            logger.error(f"Erreur chargement: {e}")
            raise
    
    def get_model_summary(self) -> Dict:
        """Retourne un résumé du modèle"""
        if self.best_model is None:
            return {'message': 'Aucun modèle entraîné'}
        
        summary = {
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'parameters': self.best_model.get_params(),
            'feature_count': len(self.feature_importance) if self.feature_importance is not None else 0
        }
        
        if self.cv_results:
            summary['available_models'] = list(self.cv_results.keys())
            summary['scores_comparison'] = {
                name: result.get('cv_mean', 0) 
                for name, result in self.cv_results.items() 
                if 'error' not in result
            }
        
        return summary
    
    def predict_proba_with_confidence(self, X: pd.DataFrame, confidence_threshold: float = 0.8) -> Dict:
        """Prédictions avec indication de confiance"""
        if self.best_model is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        try:
            predictions, probabilities = self.predict(X)
            
            if probabilities is None:
                return {
                    'predictions': predictions,
                    'confidence': None,
                    'high_confidence_mask': None
                }
            
            # Calcul de la confiance (probabilité maximale)
            max_proba = np.max(probabilities, axis=1)
            high_confidence = max_proba >= confidence_threshold
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': max_proba,
                'high_confidence_mask': high_confidence,
                'high_confidence_count': np.sum(high_confidence),
                'total_predictions': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Erreur prédiction avec confiance: {e}")
            raise

class ModelComparison:
    """Classe pour comparer plusieurs modèles"""
    
    def __init__(self, predictors: List[ElectionPredictor]):
        self.predictors = predictors
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Compare les performances de plusieurs modèles"""
        results = []
        
        for i, predictor in enumerate(self.predictors):
            if predictor.best_model is None:
                continue
                
            try:
                evaluation = predictor.evaluate_model(X_test, y_test)
                
                results.append({
                    'model_id': i,
                    'model_name': predictor.best_model_name,
                    'accuracy': evaluation.get('accuracy', 0),
                    'auc_score': evaluation.get('auc_score', 0),
                    'cv_score': predictor.best_score
                })
                
            except Exception as e:
                logger.error(f"Erreur comparaison modèle {i}: {e}")
        
        return pd.DataFrame(results)
    
    def ensemble_predict(self, X: pd.DataFrame, method: str = 'majority') -> np.ndarray:
        """Prédiction d'ensemble"""
        if not self.predictors:
            raise ValueError("Aucun modèle disponible")
        
        all_predictions = []
        
        for predictor in self.predictors:
            if predictor.best_model is not None:
                try:
                    pred, _ = predictor.predict(X)
                    all_predictions.append(pred)
                except Exception as e:
                    logger.error(f"Erreur prédiction ensemble: {e}")
        
        if not all_predictions:
            raise ValueError("Aucune prédiction disponible")
        
        predictions_array = np.array(all_predictions)
        
        if method == 'majority':
            # Vote majoritaire
            from scipy import stats
            ensemble_pred = stats.mode(predictions_array, axis=0)[0].flatten()
        else:
            # Moyenne (pour les prédictions numériques)
            ensemble_pred = np.mean(predictions_array, axis=0)
        
        return ensemble_pred