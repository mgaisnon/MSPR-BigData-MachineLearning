import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class ElectionDataProcessor:
    """Processeur de données électorales basé sur votre structure BDD"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def process_election_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traite les résultats électoraux bruts"""
        processed_df = df.copy()
        
        # Calcul des indicateurs électoraux
        processed_df['taux_participation'] = processed_df['votants'] / processed_df['inscrits']
        processed_df['taux_abstention'] = processed_df['abstentions'] / processed_df['inscrits']
        processed_df['part_voix'] = processed_df['voix'] / processed_df['exprimes']
        
        # Gestion des valeurs manquantes et aberrantes
        processed_df = processed_df.fillna(0)
        
        # Correction des valeurs > 1 pour les taux
        for col in ['taux_participation', 'taux_abstention', 'part_voix']:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].clip(0, 1)
        
        return processed_df
    
    def create_department_summary(self, df: pd.DataFrame, annee: int, tour: int) -> pd.DataFrame:
        """Crée un résumé par département pour une élection donnée"""
        election_data = df[(df['annee'] == annee) & (df['tour'] == tour)].copy()
        
        if election_data.empty:
            return pd.DataFrame()
        
        dept_summary = election_data.groupby('departement').agg({
            'inscrits': 'sum',
            'votants': 'sum',
            'abstentions': 'sum',
            'exprimes': 'sum'
        }).reset_index()
        
        # Calcul des indicateurs
        dept_summary['taux_participation'] = dept_summary['votants'] / dept_summary['inscrits']
        dept_summary['taux_abstention'] = dept_summary['abstentions'] / dept_summary['inscrits']
        
        return dept_summary
    
    def get_winner_by_department(self, df: pd.DataFrame, annee: int, tour: int) -> pd.DataFrame:
        """Détermine le candidat/parti gagnant par département"""
        election_data = df[(df['annee'] == annee) & (df['tour'] == tour)].copy()
        
        if election_data.empty:
            return pd.DataFrame()
        
        # Trouver la nuance avec le plus de voix par département
        winners = election_data.loc[election_data.groupby('departement')['voix'].idxmax()]
        
        result = winners[['departement', 'nuance', 'voix']].copy()
        result.columns = ['departement', 'nuance_gagnante', 'voix_gagnantes']
        
        return result
    
    def create_comparison_matrix(self, df: pd.DataFrame, nuances: List[str] = None) -> pd.DataFrame:
        """Crée une matrice de comparaison entre les nuances politiques"""
        if nuances is None:
            top_nuances = df['nuance'].value_counts().head(10).index.tolist()
        else:
            top_nuances = nuances
        
        # Pivot table avec les résultats par département et nuance
        try:
            pivot_df = df[df['nuance'].isin(top_nuances)].pivot_table(
                index='departement',
                columns='nuance',
                values='part_voix',
                aggfunc='mean',
                fill_value=0
            )
            return pivot_df
        except Exception as e:
            logger.error(f"Erreur création matrice: {e}")
            return pd.DataFrame()
    
    def prepare_ml_features(self, df: pd.DataFrame, target_nuance: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les features pour le machine learning"""
        try:
            # Création des features par département
            features_df = df.groupby(['departement', 'annee', 'tour']).agg({
                'taux_participation': 'mean',
                'taux_abstention': 'mean',
                'inscrits': 'sum',
                'votants': 'sum'
            }).reset_index()
            
            # Ajout des parts de voix des principales nuances
            main_nuances = df['nuance'].value_counts().head(5).index.tolist()
            
            for nuance in main_nuances:
                nuance_data = df[df['nuance'] == nuance].groupby(['departement', 'annee', 'tour']).agg({
                    'part_voix': 'mean'
                }).reset_index()
                nuance_data.columns = ['departement', 'annee', 'tour', f'part_{nuance}']
                
                features_df = features_df.merge(
                    nuance_data, 
                    on=['departement', 'annee', 'tour'], 
                    how='left'
                )
            
            # Remplir les valeurs manquantes
            features_df = features_df.fillna(0)
            
            # Définir la target
            if target_nuance:
                target_col = f'part_{target_nuance}'
                if target_col in features_df.columns:
                    y = features_df[target_col]
                else:
                    raise ValueError(f"Nuance {target_nuance} non trouvée")
            else:
                # Prédire la nuance gagnante
                unique_elections = df[['annee', 'tour']].drop_duplicates()
                if not unique_elections.empty:
                    annee, tour = unique_elections.iloc[0]['annee'], unique_elections.iloc[0]['tour']
                    winners = self.get_winner_by_department(df, annee, tour)
                    features_df = features_df.merge(winners, on='departement', how='left')
                    y = features_df['nuance_gagnante'].fillna('UNKNOWN')
                    
                    # Encoder la target
                    if 'winner' not in self.label_encoders:
                        self.label_encoders['winner'] = LabelEncoder()
                        y = pd.Series(self.label_encoders['winner'].fit_transform(y), index=y.index)
                    else:
                        y = pd.Series(self.label_encoders['winner'].transform(y), index=y.index)
                else:
                    # Target par défaut
                    y = pd.Series(np.zeros(len(features_df)), index=features_df.index)
            
            # Sélectionner les features numériques
            feature_cols = [col for col in features_df.columns if col not in 
                           ['departement', 'annee', 'tour', 'nuance_gagnante', 'voix_gagnantes']]
            
            X = features_df[feature_cols]
            
            # Standardisation
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Erreur préparation ML: {e}")
            return pd.DataFrame(), pd.Series()
    
    def get_evolution_trends(self, df: pd.DataFrame, nuance: str) -> pd.DataFrame:
        """Analyse l'évolution d'une nuance politique dans le temps"""
        try:
            evolution = df[df['nuance'] == nuance].groupby(['departement', 'annee']).agg({
                'part_voix': 'mean',
                'voix': 'sum'
            }).reset_index()
            
            if evolution.empty:
                return pd.DataFrame()
            
            # Calcul de l'évolution par rapport à l'élection précédente
            evolution = evolution.sort_values(['departement', 'annee'])
            evolution['evolution_part'] = evolution.groupby('departement')['part_voix'].pct_change()
            evolution['evolution_voix'] = evolution.groupby('departement')['voix'].pct_change()
            
            return evolution
            
        except Exception as e:
            logger.error(f"Erreur évolution trends: {e}")
            return pd.DataFrame()
    
    def analyze_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyse les corrélations entre variables"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            return corr_matrix
        except Exception as e:
            logger.error(f"Erreur analyse corrélations: {e}")
            return pd.DataFrame()
    
    def get_top_performers(self, df: pd.DataFrame, metric: str = 'voix', n: int = 10) -> pd.DataFrame:
        """Identifie les top performers selon une métrique"""
        try:
            if metric not in df.columns:
                return pd.DataFrame()
            
            top = df.nlargest(n, metric)[['departement', 'annee', 'tour', 'nuance', metric]]
            return top
            
        except Exception as e:
            logger.error(f"Erreur top performers: {e}")
            return pd.DataFrame()