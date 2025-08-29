import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional
import logging
from config.config import DATA_CONFIG

logger = logging.getLogger(__name__)

class ElectionDataProcessor:
    """Processeur de données électorales original (pour APIs externes)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fusionne les différents datasets"""
        try:
            # Identifier le dataset principal
            main_dataset = None
            for key in datasets.keys():
                if 'presidentielle' in key and '2022' in key:
                    main_dataset = key
                    break
            
            if main_dataset is None:
                main_dataset = list(datasets.keys())[0]
            
            base_df = datasets[main_dataset].copy()
            logger.info(f"Dataset principal: {main_dataset} ({len(base_df)} lignes)")
            
            # Ajout des autres datasets
            for key, df in datasets.items():
                if key != main_dataset and not df.empty:
                    if 'socio' in key:
                        # Fusion des données socio-économiques
                        base_df = self._merge_socio_data(base_df, df)
                    elif 'legislative' in key:
                        # Ajout des données législatives
                        base_df = self._merge_legislative_data(base_df, df)
            
            logger.info(f"Dataset fusionné: {len(base_df)} lignes, {len(base_df.columns)} colonnes")
            return base_df
            
        except Exception as e:
            logger.error(f"Erreur lors de la fusion: {e}")
            return pd.DataFrame()
    
    def _merge_socio_data(self, base_df: pd.DataFrame, socio_df: pd.DataFrame) -> pd.DataFrame:
        """Fusionne les données socio-économiques"""
        try:
            # Mapping des colonnes selon la source
            socio_cols_mapping = {
                'DEP': 'departement',
                'COM': 'commune',
                'LIBCOM': 'nom_commune'
            }
            
            socio_processed = socio_df.rename(columns=socio_cols_mapping)
            
            # Agrégation par département
            if 'departement' in socio_processed.columns:
                socio_agg = socio_processed.groupby('departement').agg({
                    'taux_pauvrete': 'mean',
                    'revenu_median': 'mean',
                    'taux_chomage': 'mean',
                    'part_cadres': 'mean',
                    'part_ouvriers': 'mean',
                    'age_moyen': 'mean',
                    'densite_population': 'mean'
                }).reset_index()
                
                # Fusion avec le dataset principal
                base_df = base_df.merge(socio_agg, on='departement', how='left')
            
            return base_df
            
        except Exception as e:
            logger.error(f"Erreur fusion socio: {e}")
            return base_df
    
    def _merge_legislative_data(self, base_df: pd.DataFrame, leg_df: pd.DataFrame) -> pd.DataFrame:
        """Fusionne les données législatives"""
        try:
            # Agrégation des résultats législatives par département
            if len(leg_df) > 0:
                leg_agg = leg_df.groupby('departement').agg({
                    'inscrits': 'mean',
                    'votants': 'mean',
                    'abstentions': 'mean'
                }).reset_index()
                
                leg_agg.columns = ['departement', 'inscrits_leg', 'votants_leg', 'abstentions_leg']
                base_df = base_df.merge(leg_agg, on='departement', how='left')
            
            return base_df
            
        except Exception as e:
            logger.error(f"Erreur fusion législatives: {e}")
            return base_df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des variables dérivées"""
        df_processed = df.copy()
        
        try:
            # Variables électorales de base
            if all(col in df_processed.columns for col in ['votants', 'inscrits']):
                df_processed['participation_rate'] = df_processed['votants'] / df_processed['inscrits']
                df_processed['abstention_rate'] = 1 - df_processed['participation_rate']
            
            # Variables spécifiques aux présidentielles
            if 'voix' in df_processed.columns:
                df_processed['vote_share'] = df_processed['voix'] / df_processed.get('exprimes', df_processed['voix'])
            
            # Variables socio-économiques dérivées
            if 'taux_pauvrete' in df_processed.columns:
                df_processed['niveau_precarite'] = pd.cut(
                    df_processed['taux_pauvrete'], 
                    bins=[0, 10, 20, 100], 
                    labels=['Faible', 'Moyen', 'Élevé']
                )
            
            if 'revenu_median' in df_processed.columns:
                df_processed['niveau_revenu'] = pd.cut(
                    df_processed['revenu_median'],
                    bins=[0, 20000, 30000, 100000],
                    labels=['Bas', 'Moyen', 'Élevé']
                )
            
            # Variables géographiques
            if 'densite_population' in df_processed.columns:
                df_processed['type_territoire'] = pd.cut(
                    df_processed['densite_population'],
                    bins=[0, 50, 150, 1000000],
                    labels=['Rural', 'Péri-urbain', 'Urbain']
                )
            
            logger.info(f"Features créées: {len(df_processed.columns)} colonnes")
            return df_processed
            
        except Exception as e:
            logger.error(f"Erreur création features: {e}")
            return df_processed
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données"""
        df_clean = df.copy()
        
        try:
            # Suppression des lignes complètement vides
            df_clean = df_clean.dropna(how='all')
            
            # Suppression des doublons
            df_clean = df_clean.drop_duplicates()
            
            # Traitement des valeurs aberrantes pour les taux
            rate_columns = [col for col in df_clean.columns if 'rate' in col or 'taux' in col]
            for col in rate_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].clip(0, 1)
            
            # Imputation des valeurs manquantes
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
            
            # Traitement des outliers extrêmes
            for col in numeric_cols:
                if col not in ['annee', 'tour', 'id']:
                    Q1 = df_clean[col].quantile(0.01)
                    Q99 = df_clean[col].quantile(0.99)
                    df_clean[col] = df_clean[col].clip(Q1, Q99)
            
            logger.info(f"Données nettoyées: {len(df_clean)} lignes conservées")
            return df_clean
            
        except Exception as e:
            logger.error(f"Erreur nettoyage: {e}")
            return df_clean
    
    def prepare_features_target(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les features et la target pour le ML"""
        try:
            # Définition des colonnes features
            feature_cols = [
                'participation_rate', 'taux_pauvrete', 'revenu_median',
                'taux_chomage', 'part_cadres', 'part_ouvriers', 
                'age_moyen', 'densite_population'
            ]
            
            # Filtrer les colonnes existantes
            available_features = [col for col in feature_cols if col in df.columns]
            
            if not available_features:
                raise ValueError("Aucune feature disponible")
            
            X = df[available_features].copy()
            
            # Définition de la target
            if target_column and target_column in df.columns:
                y = df[target_column].copy()
            else:
                # Créer une target par défaut (candidat majoritaire simulé)
                y = np.random.choice(['Candidat_A', 'Candidat_B'], size=len(df))
                y = pd.Series(y, index=df.index)
            
            # Encodage de la variable cible si nécessaire
            if y.dtype == 'object':
                if target_column not in self.label_encoders:
                    self.label_encoders[target_column or 'target'] = LabelEncoder()
                    y_encoded = self.label_encoders[target_column or 'target'].fit_transform(y)
                else:
                    y_encoded = self.label_encoders[target_column or 'target'].transform(y)
                y = pd.Series(y_encoded, index=y.index)
            
            # Standardisation des features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            logger.info(f"Features préparées: {X_scaled.shape}, Target: {y.shape}")
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Erreur préparation ML: {e}")
            return pd.DataFrame(), pd.Series()
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Sauvegarde les données traitées"""
        try:
            filepath = DATA_CONFIG.processed_data_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Données sauvegardées: {filepath}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Charge les données traitées"""
        try:
            filepath = DATA_CONFIG.processed_data_dir / filename
            df = pd.read_csv(filepath)
            logger.info(f"Données chargées: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Erreur chargement: {e}")
            return pd.DataFrame()