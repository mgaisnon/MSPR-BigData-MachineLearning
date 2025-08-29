import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration du chemin
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

logger = logging.getLogger(__name__)

class ElectionPredictor:
    """Classe pour les prédictions électorales avec Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.best_model_name = None
        self.best_score = 0
        self._last_results = {}
        
        logger.info("ElectionPredictor initialisé")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les features pour le Machine Learning"""
        try:
            logger.info(f"Préparation des features pour {len(df)} enregistrements")
            
            data = df.copy()
            
            # Définir le target (ce qu'on veut prédire)
            if 'famille_politique' in data.columns:
                target = data['famille_politique'].copy()
                logger.info("Target: famille_politique")
            elif 'nuance' in data.columns:
                target = data['nuance'].copy()
                logger.info("Target: nuance")
            else:
                raise ValueError("Pas de colonne target trouvée")
            
            # Créer le DataFrame des features
            features_df = pd.DataFrame()
            
            # Features temporelles
            if 'annee' in data.columns:
                features_df['annee'] = data['annee']
                features_df['decennie'] = (data['annee'] // 10) * 10
                features_df['depuis_2000'] = (data['annee'] - 2000).clip(lower=0)
            
            # Features géographiques
            if 'departement' in data.columns:
                dept_encoder = LabelEncoder()
                features_df['departement_encoded'] = dept_encoder.fit_transform(data['departement'].astype(str))
                self.encoders['departement'] = dept_encoder
            
            if 'typologie' in data.columns:
                features_df['is_urbain'] = (data['typologie'] == 'Urbain').astype(int)
            
            if 'ancien_midi_pyrenees' in data.columns:
                features_df['ancien_midi_pyrenees'] = data['ancien_midi_pyrenees'].astype(int)
            
            # Features électorales (numériques)
            numeric_features = ['taux_participation', 'taux_abstention', 'voix', 'inscrits', 'votants']
            
            for feature in numeric_features:
                if feature in data.columns:
                    features_df[feature] = pd.to_numeric(data[feature], errors='coerce').fillna(0)
            
            # Features dérivées
            if all(col in features_df.columns for col in ['voix', 'inscrits']):
                features_df['influence'] = features_df['voix'] / (features_df['inscrits'] + 1)
            
            if 'taux_participation' in features_df.columns:
                features_df['participation_haute'] = (features_df['taux_participation'] > 70).astype(int)
                features_df['participation_faible'] = (features_df['taux_participation'] < 50).astype(int)
            
            # Tour (si disponible)
            if 'tour' in data.columns:
                features_df['tour'] = data['tour']
            
            # Nettoyer les données
            features_df = features_df.fillna(0)
            
            # Encoder le target
            target_encoder = LabelEncoder()
            target_encoded = target_encoder.fit_transform(target.astype(str).fillna('Inconnu'))
            self.encoders['target'] = target_encoder
            
            logger.info(f"Features créées: {list(features_df.columns)}")
            logger.info(f"Nombre de classes target: {len(target_encoder.classes_)}")
            logger.info(f"Échantillons finaux: {len(features_df)}")
            
            return features_df, pd.Series(target_encoded)
            
        except Exception as e:
            logger.error(f"Erreur préparation features: {e}")
            # Fallback simple
            fallback_df = pd.DataFrame({
                'dummy_feature': range(len(df)),
                'random_feature': np.random.random(len(df))
            })
            fallback_target = pd.Series([0] * len(df))
            return fallback_df, fallback_target
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entraîne les modèles de Machine Learning"""
        try:
            logger.info("🤖 Début de l'entraînement des modèles ML")
            
            # Vérifications préliminaires
            if len(X) < 20:
                return {"error": "Pas assez de données pour l'entraînement (minimum 20)"}
            
            unique_classes = len(np.unique(y))
            if unique_classes < 2:
                return {"error": f"Pas assez de classes différentes ({unique_classes})"}
            
            logger.info(f"Données: {len(X)} échantillons, {len(X.columns)} features, {unique_classes} classes")
            
            # Division train/test
            test_size = min(0.3, max(0.1, 50/len(X)))  # Adaptatif selon la taille
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Configuration des modèles
            models_to_train = {
                'RandomForest': {
                    'model': RandomForestClassifier(
                        n_estimators=100, 
                        max_depth=10,
                        random_state=42, 
                        n_jobs=-1
                    ),
                    'needs_scaling': False
                },
                'LogisticRegression': {
                    'model': LogisticRegression(
                        random_state=42, 
                        max_iter=1000,
                        C=1.0
                    ),
                    'needs_scaling': True
                }
            }
            
            results = {}
            
            # Entraînement de chaque modèle
            for model_name, config in models_to_train.items():
                try:
                    logger.info(f"⚡ Entraînement {model_name}...")
                    
                    model = config['model']
                    
                    # Gestion du scaling pour la régression logistique
                    if config['needs_scaling']:
                        scaler = StandardScaler()
                        X_train_processed = scaler.fit_transform(X_train)
                        X_test_processed = scaler.transform(X_test)
                        self.scalers[model_name] = scaler
                    else:
                        X_train_processed = X_train
                        X_test_processed = X_test
                    
                    # Entraînement
                    model.fit(X_train_processed, y_train)
                    
                    # Prédictions
                    y_pred = model.predict(X_test_processed)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Stockage du modèle
                    self.models[model_name] = model
                    
                    # Feature importance (pour RandomForest)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        feature_importance = importance_df.to_dict('records')
                    
                    # Résultats
                    results[model_name] = {
                        'accuracy': accuracy,
                        'n_train': len(X_train),
                        'n_test': len(X_test),
                        'n_features': len(X.columns),
                        'n_classes': unique_classes,
                        'feature_importance': feature_importance
                    }
                    
                    # Suivre le meilleur modèle
                    if accuracy > self.best_score:
                        self.best_score = accuracy
                        self.best_model_name = model_name
                    
                    logger.info(f"✅ {model_name}: Accuracy = {accuracy:.4f}")
                    
                except Exception as model_error:
                    logger.error(f"❌ Erreur {model_name}: {model_error}")
                    results[model_name] = {'error': str(model_error)}
            
            # Sauvegarde des résultats
            self._last_results = results
            
            # Sauvegarde des modèles
            self.save_models()
            
            logger.info(f"🏆 Meilleur modèle: {self.best_model_name} (accuracy: {self.best_score:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur générale entraînement: {e}")
            return {"error": str(e)}
    
    def predict_election(self, features: Dict, model_name: str = None) -> Dict:
        """Fait une prédiction pour de nouvelles données"""
        try:
            if not self.models:
                return {"error": "Aucun modèle entraîné disponible"}
            
            # Utiliser le meilleur modèle par défaut
            model_name = model_name or self.best_model_name or list(self.models.keys())[0]
            
            if model_name not in self.models:
                return {"error": f"Modèle {model_name} non trouvé"}
            
            model = self.models[model_name]
            
            # Préparer les données
            features_df = pd.DataFrame([features])
            
            # Appliquer le scaling si nécessaire
            if model_name in self.scalers:
                features_processed = self.scalers[model_name].transform(features_df)
            else:
                features_processed = features_df
            
            # Prédiction
            prediction = model.predict(features_processed)[0]
            
            # Probabilités (si disponible)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_processed)[0]
                if 'target' in self.encoders:
                    classes = self.encoders['target'].classes_
                    probabilities = {classes[i]: float(proba[i]) for i in range(len(classes))}
            
            # Décoder la prédiction
            if 'target' in self.encoders:
                predicted_class = self.encoders['target'].inverse_transform([prediction])[0]
            else:
                predicted_class = str(prediction)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': float(max(proba)) if probabilities else 0.5,
                'probabilities': probabilities,
                'model_used': model_name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            return {"error": str(e)}
    
    def save_models(self):
        """Sauvegarde tous les modèles entraînés"""
        try:
            # Import de la config
            from config.config import MODEL_CONFIG
            models_dir = MODEL_CONFIG.models_dir
            models_dir.mkdir(exist_ok=True, parents=True)
            
            # Sauvegarder chaque modèle
            for model_name, model in self.models.items():
                model_file = models_dir / f"{model_name.lower()}_model.joblib"
                joblib.dump(model, model_file)
                logger.info(f"💾 Modèle {model_name} sauvegardé: {model_file}")
            
            # Sauvegarder les scalers
            if self.scalers:
                scalers_file = models_dir / "scalers.joblib"
                joblib.dump(self.scalers, scalers_file)
                logger.info(f"💾 Scalers sauvegardés: {scalers_file}")
            
            # Sauvegarder les encoders
            if self.encoders:
                encoders_file = models_dir / "encoders.joblib"
                joblib.dump(self.encoders, encoders_file)
                logger.info(f"💾 Encoders sauvegardés: {encoders_file}")
            
            # Métadonnées
            metadata = {
                'best_model_name': self.best_model_name,
                'best_score': self.best_score,
                'models_list': list(self.models.keys()),
                'last_results': self._last_results
            }
            metadata_file = models_dir / "metadata.joblib"
            joblib.dump(metadata, metadata_file)
            logger.info(f"💾 Métadonnées sauvegardées: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
    
    def load_models(self):
        """Charge les modèles précédemment sauvegardés"""
        try:
            from config.config import MODEL_CONFIG
            models_dir = MODEL_CONFIG.models_dir
            
            metadata_file = models_dir / "metadata.joblib"
            if not metadata_file.exists():
                logger.warning("Aucune sauvegarde de modèles trouvée")
                return False
            
            # Charger les métadonnées
            metadata = joblib.load(metadata_file)
            self.best_model_name = metadata.get('best_model_name')
            self.best_score = metadata.get('best_score', 0)
            self._last_results = metadata.get('last_results', {})
            
            # Charger chaque modèle
            for model_name in metadata.get('models_list', []):
                model_file = models_dir / f"{model_name.lower()}_model.joblib"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"📂 Modèle {model_name} chargé")
            
            # Charger scalers et encoders
            scalers_file = models_dir / "scalers.joblib"
            if scalers_file.exists():
                self.scalers = joblib.load(scalers_file)
                logger.info("📂 Scalers chargés")
            
            encoders_file = models_dir / "encoders.joblib"
            if encoders_file.exists():
                self.encoders = joblib.load(encoders_file)
                logger.info("📂 Encoders chargés")
            
            logger.info("✅ Tous les modèles chargés avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement: {e}")
            return False
    
    def get_model_performance(self) -> Dict:
        """Retourne les performances des modèles"""
        return self._last_results
    
    def get_available_models(self) -> list:
        """Retourne la liste des modèles disponibles"""
        return list(self.models.keys())

# Fonction utilitaire pour l'intégration avec main.py
def train_election_models(processed_data_file: str = None):
    """
    Fonction principale pour entraîner les modèles
    Utilisée par main.py dans l'étape prédictions
    """
    try:
        from config.config import DATA_CONFIG
        
        # Fichier par défaut
        if processed_data_file is None:
            processed_data_file = DATA_CONFIG.processed_data_dir / 'elections_processed.csv'
        
        # Vérifier l'existence du fichier
        if not Path(processed_data_file).exists():
            return None, {"error": f"Fichier non trouvé: {processed_data_file}"}
        
        # Charger les données
        data = pd.read_csv(processed_data_file)
        logger.info(f"📊 Données chargées: {len(data)} enregistrements")
        
        # Vérifications de base
        if len(data) < 20:
            return None, {"error": "Pas assez de données (minimum 20 enregistrements)"}
        
        # Créer et configurer le prédicteur
        predictor = ElectionPredictor()
        
        # Préparer les features
        X, y = predictor.prepare_features(data)
        
        # Entraîner les modèles
        if len(X) >= 20 and len(np.unique(y)) >= 2:
            results = predictor.train_models(X, y)
            return predictor, results
        else:
            return predictor, {"error": "Données insuffisantes pour l'entraînement ML"}
            
    except Exception as e:
        logger.error(f"Erreur train_election_models: {e}")
        return None, {"error": str(e)}

# Test du module (si lancé directement)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🧪 TEST DU MODULE MACHINE LEARNING")
    print("=" * 50)
    
    # Test avec des données fictives
    test_data = pd.DataFrame({
        'annee': [2017, 2017, 2022, 2022] * 10,
        'departement': [30, 31, 30, 31] * 10,
        'famille_politique': ['Droite', 'Gauche', 'Centre', 'Droite'] * 10,
        'voix': [1000, 1500, 800, 1200] * 10,
        'inscrits': [2000, 2000, 2000, 2000] * 10,
        'taux_participation': [65.5, 70.2, 60.1, 75.3] * 10,
        'typologie': ['Urbain', 'Rural', 'Urbain', 'Rural'] * 10,
        'ancien_midi_pyrenees': [0, 1, 0, 1] * 10
    })
    
    print(f"Données test: {len(test_data)} enregistrements")
    
    predictor = ElectionPredictor()
    X, y = predictor.prepare_features(test_data)
    print(f"Features: {X.shape}, Target: {y.shape}")
    
    results = predictor.train_models(X, y)
    print("\nRésultats:")
    for model_name, result in results.items():
        if 'accuracy' in result:
            print(f"  {model_name}: {result['accuracy']:.4f}")
        else:
            print(f"  {model_name}: {result}")
    
    print("\n✅ Test terminé")