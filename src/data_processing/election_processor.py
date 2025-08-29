import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import sys

# Ajout du répertoire parent pour les imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from config.config import OCCITANIE_CONFIG

logger = logging.getLogger(__name__)

class ElectionDataProcessor:
    """Processeur de données électorales pour l'Occitanie"""
    
    def __init__(self):
        self.occitanie_config = OCCITANIE_CONFIG
        logger.info("ElectionDataProcessor initialisé")
    
    def preprocess_election_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Traite et nettoie les données électorales brutes"""
        try:
            logger.info(f"Début du traitement de {len(df_raw)} enregistrements bruts")
            
            if df_raw.empty:
                logger.warning("DataFrame vide en entrée")
                return pd.DataFrame()
            
            # Copie pour éviter les modifications sur l'original
            df = df_raw.copy()
            
            # Étape 1: Nettoyage des colonnes
            df = self._clean_columns(df)
            logger.info(f"Après nettoyage des colonnes: {len(df)} enregistrements")
            
            # Étape 2: Validation des données
            df = self._validate_data(df)
            logger.info(f"Après validation: {len(df)} enregistrements")
            
            # Étape 3: Enrichissement des données
            df = self._enrich_data(df)
            logger.info(f"Après enrichissement: {len(df)} enregistrements")
            
            # Étape 4: Calculs des indicateurs
            df = self._calculate_indicators(df)
            logger.info(f"Après calculs: {len(df)} enregistrements")
            
            # Étape 5: Nettoyage final
            df = self._final_cleaning(df)
            logger.info(f"Traitement terminé: {len(df)} enregistrements finaux")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur dans preprocess_election_data: {e}")
            # En cas d'erreur, retourner les données brutes nettoyées minimalement
            return self._minimal_cleaning(df_raw)
    
    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les noms de colonnes et les types"""
        try:
            # Standardisation des noms de colonnes
            column_mapping = {
                'année': 'annee',
                'Année': 'annee',
                'ANNEE': 'annee',
                'Tour': 'tour',
                'TOUR': 'tour',
                'Departement': 'departement',
                'DEPARTEMENT': 'departement',
                'Nuance': 'nuance',
                'NUANCE': 'nuance',
                'Voix': 'voix',
                'VOIX': 'voix',
                'Inscrits': 'inscrits',
                'INSCRITS': 'inscrits',
                'Votants': 'votants',
                'VOTANTS': 'votants',
                'Abstentions': 'abstentions',
                'ABSTENTIONS': 'abstentions',
                'Exprimes': 'exprimes',
                'EXPRIMES': 'exprimes'
            }
            
            # Renommer les colonnes
            df = df.rename(columns=column_mapping)
            
            # Conversion des types
            if 'annee' in df.columns:
                df['annee'] = pd.to_numeric(df['annee'], errors='coerce')
            if 'tour' in df.columns:
                df['tour'] = pd.to_numeric(df['tour'], errors='coerce')
            if 'departement' in df.columns:
                df['departement'] = df['departement'].astype(str).str.zfill(2)
            
            # Colonnes numériques
            numeric_cols = ['voix', 'inscrits', 'votants', 'abstentions', 'exprimes']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Nettoyage des nuances politiques
            if 'nuance' in df.columns:
                df['nuance'] = df['nuance'].astype(str).str.strip().str.upper()
                df['nuance'] = df['nuance'].replace({'NAN': 'AUTRE', '': 'AUTRE', 'NULL': 'AUTRE'})
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur nettoyage colonnes: {e}")
            return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide la cohérence des données"""
        try:
            initial_count = len(df)
            
            # Supprimer les lignes avec des valeurs manquantes critiques
            if 'annee' in df.columns:
                df = df[df['annee'].notna()]
            if 'departement' in df.columns:
                df = df[df['departement'].notna()]
            if 'tour' in df.columns:
                df = df[df['tour'].notna()]
            
            # Filtrer sur les départements d'Occitanie
            if 'departement' in df.columns:
                df = df[df['departement'].isin(self.occitanie_config.all_departments)]
            
            # Années cohérentes (1990-2025)
            if 'annee' in df.columns:
                df = df[(df['annee'] >= 1990) & (df['annee'] <= 2025)]
            
            # Tours valides (1 ou 2)
            if 'tour' in df.columns:
                df = df[df['tour'].isin([1, 2])]
            
            # Voix cohérentes (>= 0)
            numeric_cols = ['voix', 'inscrits', 'votants', 'abstentions', 'exprimes']
            for col in numeric_cols:
                if col in df.columns:
                    df = df[df[col] >= 0]
            
            logger.info(f"Validation: {initial_count} -> {len(df)} enregistrements")
            return df
            
        except Exception as e:
            logger.error(f"Erreur validation: {e}")
            return df
    
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit les données avec des informations supplémentaires"""
        try:
            # Ajouter les noms des départements
            if 'departement' in df.columns:
                df['departement_nom'] = df['departement'].map(
                    self.occitanie_config.department_names
                ).fillna('Inconnu')
            
            # Catégorisation des nuances politiques
            if 'nuance' in df.columns:
                df['famille_politique'] = df['nuance'].apply(self._categorize_nuance)
            
            # Indicateur ancien/nouveau département Occitanie
            if 'departement' in df.columns:
                df['ancien_midi_pyrenees'] = df['departement'].isin(
                    self.occitanie_config.midi_pyrenees_depts
                )
                df['ancien_languedoc_roussillon'] = df['departement'].isin(
                    self.occitanie_config.languedoc_roussillon_depts
                )
            
            # Classification urbain/rural (approximative)
            if 'departement' in df.columns:
                urbain_depts = ['31', '34', '30']  # Toulouse, Montpellier, Nîmes
                df['typologie'] = df['departement'].apply(
                    lambda x: 'Urbain' if x in urbain_depts else 'Rural'
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur enrichissement: {e}")
            return df
    
    def _categorize_nuance(self, nuance: str) -> str:
        """Catégorise les nuances politiques en familles"""
        if pd.isna(nuance) or nuance == 'AUTRE':
            return 'Autres'
        
        nuance = str(nuance).upper()
        
        # Extrême droite
        if any(x in nuance for x in ['RN', 'FN', 'DLF', 'REC']):
            return 'Extrême droite'
        
        # Droite
        if any(x in nuance for x in ['LR', 'UMP', 'RPR', 'UDI', 'DVD', 'DLR']):
            return 'Droite'
        
        # Centre
        if any(x in nuance for x in ['REN', 'MODEM', 'UDI', 'LREM', 'EM']):
            return 'Centre'
        
        # Gauche
        if any(x in nuance for x in ['PS', 'PRG', 'DVG', 'MRG']):
            return 'Gauche'
        
        # Extrême gauche
        if any(x in nuance for x in ['LFI', 'PCF', 'LO', 'NPA', 'POI']):
            return 'Extrême gauche'
        
        # Écologie
        if any(x in nuance for x in ['EELV', 'VEC', 'GEN']):
            return 'Écologie'
        
        return 'Autres'
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule des indicateurs électoraux"""
        try:
            # Taux de participation
            if all(col in df.columns for col in ['votants', 'inscrits']):
                df['taux_participation'] = np.where(
                    df['inscrits'] > 0,
                    (df['votants'] / df['inscrits'] * 100).round(2),
                    0
                )
            
            # Taux d'abstention
            if all(col in df.columns for col in ['abstentions', 'inscrits']):
                df['taux_abstention'] = np.where(
                    df['inscrits'] > 0,
                    (df['abstentions'] / df['inscrits'] * 100).round(2),
                    0
                )
            
            # Pourcentage des voix sur exprimés
            if all(col in df.columns for col in ['voix', 'exprimes']):
                df['pct_exprimes'] = np.where(
                    df['exprimes'] > 0,
                    (df['voix'] / df['exprimes'] * 100).round(2),
                    0
                )
            
            # Pourcentage des voix sur inscrits
            if all(col in df.columns for col in ['voix', 'inscrits']):
                df['pct_inscrits'] = np.where(
                    df['inscrits'] > 0,
                    (df['voix'] / df['inscrits'] * 100).round(2),
                    0
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calculs indicateurs: {e}")
            return df
    
    def _final_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage final et réorganisation"""
        try:
            # Supprimer les doublons
            initial_count = len(df)
            if 'annee' in df.columns and 'tour' in df.columns and 'departement' in df.columns and 'nuance' in df.columns:
                df = df.drop_duplicates(subset=['annee', 'tour', 'departement', 'nuance'])
                logger.info(f"Suppression doublons: {initial_count} -> {len(df)}")
            
            # Réorganiser les colonnes
            base_cols = ['id', 'annee', 'tour', 'departement', 'departement_nom']
            data_cols = ['inscrits', 'votants', 'abstentions', 'exprimes', 'voix']
            indicator_cols = ['taux_participation', 'taux_abstention', 'pct_exprimes', 'pct_inscrits']
            political_cols = ['nuance', 'famille_politique']
            geographic_cols = ['ancien_midi_pyrenees', 'ancien_languedoc_roussillon', 'typologie']
            
            # Colonnes à garder (seulement celles qui existent)
            columns_order = []
            for col_group in [base_cols, political_cols, data_cols, indicator_cols, geographic_cols]:
                for col in col_group:
                    if col in df.columns:
                        columns_order.append(col)
            
            # Ajouter les colonnes restantes
            remaining_cols = [col for col in df.columns if col not in columns_order]
            columns_order.extend(remaining_cols)
            
            df = df[columns_order]
            
            # Tri final
            sort_cols = []
            if 'annee' in df.columns:
                sort_cols.append('annee')
            if 'tour' in df.columns:
                sort_cols.append('tour')
            if 'departement' in df.columns:
                sort_cols.append('departement')
            if 'voix' in df.columns:
                sort_cols.append('voix')
            
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=[True, True, True, False])
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur nettoyage final: {e}")
            return df
    
    def _minimal_cleaning(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage minimal en cas d'erreur"""
        try:
            logger.warning("Utilisation du nettoyage minimal")
            df = df_raw.copy()
            
            # Conversion basique des types
            numeric_cols = ['annee', 'tour', 'voix', 'inscrits', 'votants', 'abstentions', 'exprimes']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Nettoyage départements
            if 'departement' in df.columns:
                df['departement'] = df['departement'].astype(str).str.zfill(2)
                df = df[df['departement'].isin(self.occitanie_config.all_departments)]
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur nettoyage minimal: {e}")
            return df_raw
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Génère des statistiques de résumé"""
        try:
            stats = {
                'total_records': len(df),
                'years': sorted(df['annee'].unique().tolist()) if 'annee' in df.columns else [],
                'tours': sorted(df['tour'].unique().tolist()) if 'tour' in df.columns else [],
                'departments': sorted(df['departement'].unique().tolist()) if 'departement' in df.columns else [],
                'nuances': df['nuance'].nunique() if 'nuance' in df.columns else 0,
                'total_voix': df['voix'].sum() if 'voix' in df.columns else 0,
                'avg_participation': df['taux_participation'].mean() if 'taux_participation' in df.columns else None
            }
            return stats
        except Exception as e:
            logger.error(f"Erreur calcul statistiques: {e}")
            return {'error': str(e)}
    def train_election_models(processed_data_file: str = None):
        try:
            import sys
            from pathlib import Path
            
            # Ajout du path
            current_dir = Path(__file__).parent
            root_dir = current_dir.parent.parent
            sys.path.insert(0, str(root_dir))
            
            from config.config import DATA_CONFIG
            import logging
            
            logger = logging.getLogger(__name__)
            
            if processed_data_file is None:
                processed_data_file = DATA_CONFIG.processed_data_dir / 'elections_processed.csv'
            
            # Charger les données
            if Path(processed_data_file).exists():
                data = pd.read_csv(processed_data_file)
                logger.info(f"Données ML chargées: {len(data)} enregistrements")
                
                # Créer et entraîner le prédicteur
                predictor = ElectionPredictor()
                X, y = predictor.prepare_features(data)
                
                if len(X) > 10:
                    results = predictor.train_models(X, y)
                    return predictor, results
                else:
                    return predictor, {"error": "Pas assez de données"}
            else:
                return None, {"error": "Fichier de données non trouvé"}
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur entraînement global: {e}")
            return None, {"error": str(e)}
    
# Test du processeur
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("TEST DU PROCESSEUR DE DONNÉES")
    print("=" * 40)
    
    # Test avec des données fictives
    test_data = pd.DataFrame({
        'annee': [2022, 2022, 2017, 2017],
        'tour': [1, 2, 1, 2],
        'departement': ['31', '31', '34', '34'],
        'nuance': ['RN', 'REN', 'LFI', 'PS'],
        'voix': [1000, 1500, 800, 1200],
        'inscrits': [10000, 10000, 8000, 8000],
        'votants': [7500, 8500, 6000, 6500],
        'abstentions': [2500, 1500, 2000, 1500],
        'exprimes': [7000, 8000, 5500, 6000]
    })
    
    processor = ElectionDataProcessor()
    result = processor.preprocess_election_data(test_data)
    
    print(f"Données traitées: {len(result)} enregistrements")
    print(f"Colonnes ajoutées: {set(result.columns) - set(test_data.columns)}")
    print("\nAperçu:")
    print(result.head())
    
    stats = processor.get_summary_stats(result)
    print(f"\nStatistiques: {stats}")