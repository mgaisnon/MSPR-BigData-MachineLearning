import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
import logging
from config.config import DB_CONFIG, OCCITANIE_CONFIG
from typing import Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class ElectionDBCollector:
    """Collecteur de données depuis la base MySQL"""
    
    def __init__(self):
        self.config = DB_CONFIG
        self.occitanie_depts = OCCITANIE_CONFIG.all_departments
        self.engine = None
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Établit la connexion à la base de données"""
        try:
            # Connexion SQLAlchemy pour pandas
            self.engine = create_engine(self.config.connection_string)
            
            # Test de connexion
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Connexion à la base de données établie")
                
        except Exception as e:
            logger.error(f"Erreur de connexion à la base: {e}")
            logger.info("Création de données d'exemple...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Crée des données d'exemple si la BDD n'est pas accessible"""
        logger.info("Génération de données d'exemple")
        
        data = []
        nuances = ['REN', 'RN', 'LFI', 'LR', 'PS', 'EELV', 'PCF', 'DVG', 'DVD', 'DIV']
        
        for annee in [2017, 2022]:
            for tour in [1, 2]:
                for dept in self.occitanie_depts:
                    total_inscrits = np.random.randint(50000, 200000)
                    total_votants = int(total_inscrits * np.random.uniform(0.65, 0.85))
                    total_abstentions = total_inscrits - total_votants
                    total_exprimes = int(total_votants * np.random.uniform(0.96, 0.99))
                    
                    # Répartition des voix selon les nuances
                    voix_restantes = total_exprimes
                    
                    for i, nuance in enumerate(nuances):
                        if i == len(nuances) - 1:
                            voix = voix_restantes
                        else:
                            if nuance in ['REN', 'RN']:  # Principales forces
                                voix = int(total_exprimes * np.random.uniform(0.15, 0.30))
                            elif nuance in ['LFI', 'LR']:
                                voix = int(total_exprimes * np.random.uniform(0.08, 0.20))
                            else:
                                voix = int(total_exprimes * np.random.uniform(0.02, 0.10))
                            
                            voix = min(voix, voix_restantes)
                            voix_restantes -= voix
                        
                        if voix > 0:
                            data.append({
                                'id': len(data) + 1,
                                'annee': annee,
                                'tour': tour,
                                'departement': dept,
                                'inscrits': total_inscrits,
                                'votants': total_votants,
                                'abstentions': total_abstentions,
                                'exprimes': total_exprimes,
                                'nuance': nuance,
                                'voix': voix
                            })
        
        self.sample_data = pd.DataFrame(data)
        logger.info(f"Généré {len(self.sample_data)} enregistrements d'exemple")
    
    def get_election_data(self, annees: List[int] = None, tours: List[int] = None) -> pd.DataFrame:
        """Récupère les données électorales"""
        try:
            if self.engine is None:
                return self.sample_data if hasattr(self, 'sample_data') else pd.DataFrame()
            
            # Construction de la requête
            query = """
            SELECT 
                id,
                annee,
                tour,
                departement,
                inscrits,
                votants,
                abstentions,
                exprimes,
                nuance,
                voix
            FROM resultatslelegi
            WHERE departement IN ({})
            """.format(','.join([f"'{dept}'" for dept in self.occitanie_depts]))
            
            # Filtres optionnels
            if annees:
                annees_str = ','.join(map(str, annees))
                query += f" AND annee IN ({annees_str})"
            
            if tours:
                tours_str = ','.join(map(str, tours))
                query += f" AND tour IN ({tours_str})"
            
            query += " ORDER BY annee, tour, departement, nuance"
            
            # Exécution
            df = pd.read_sql(query, self.engine)
            logger.info(f"Récupéré {len(df)} enregistrements de la base")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération: {e}")
            return self.sample_data if hasattr(self, 'sample_data') else pd.DataFrame()
    
    def get_aggregated_results(self, annee: int, tour: int) -> pd.DataFrame:
        """Résultats agrégés par département"""
        try:
            if self.engine is None:
                df = self.sample_data
                return df[(df['annee'] == annee) & (df['tour'] == tour)].groupby(['departement', 'nuance']).agg({
                    'inscrits': 'sum',
                    'votants': 'sum',
                    'abstentions': 'sum',
                    'exprimes': 'sum',
                    'voix': 'sum'
                }).reset_index()
            
            query = """
            SELECT 
                departement,
                annee,
                tour,
                SUM(inscrits) as total_inscrits,
                SUM(votants) as total_votants,
                SUM(abstentions) as total_abstentions,
                SUM(exprimes) as total_exprimes,
                nuance,
                SUM(voix) as total_voix
            FROM resultatslelegi
            WHERE departement IN ({})
            AND annee = {}
            AND tour = {}
            GROUP BY departement, nuance
            ORDER BY departement, total_voix DESC
            """.format(
                ','.join([f"'{dept}'" for dept in self.occitanie_depts]),
                annee,
                tour
            )
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"Agrégé {len(df)} résultats pour {annee} tour {tour}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur agrégation: {e}")
            return pd.DataFrame()
    
    def get_available_elections(self) -> pd.DataFrame:
        """Liste des élections disponibles"""
        try:
            if self.engine is None:
                return self.sample_data[['annee', 'tour']].drop_duplicates().sort_values(['annee', 'tour'], ascending=[False, True])
            
            query = """
            SELECT DISTINCT annee, tour
            FROM resultatslelegi
            WHERE departement IN ({})
            ORDER BY annee DESC, tour
            """.format(','.join([f"'{dept}'" for dept in self.occitanie_depts]))
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.error(f"Erreur élections disponibles: {e}")
            return pd.DataFrame()
    
    def get_nuances_info(self) -> pd.DataFrame:
        """Informations sur les nuances politiques"""
        try:
            if self.engine is None:
                return self.sample_data['nuance'].value_counts().reset_index()
            
            query = """
            SELECT DISTINCT nuance, COUNT(*) as nb_occurrences
            FROM resultatslelegi
            WHERE departement IN ({})
            GROUP BY nuance
            ORDER BY nb_occurrences DESC
            """.format(','.join([f"'{dept}'" for dept in self.occitanie_depts]))
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.error(f"Erreur nuances: {e}")
            return pd.DataFrame()
    
    def get_department_stats(self) -> pd.DataFrame:
        """Statistiques par département"""
        try:
            data = self.get_election_data()
            if data.empty:
                return pd.DataFrame()
            
            stats = data.groupby('departement').agg({
                'inscrits': 'mean',
                'votants': 'mean',
                'abstentions': 'mean',
                'annee': 'nunique',
                'tour': 'nunique'
            }).reset_index()
            
            stats['taux_participation_moyen'] = stats['votants'] / stats['inscrits']
            stats['nom_departement'] = stats['departement'].map(OCCITANIE_CONFIG.department_names)
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur stats départements: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Ferme la connexion"""
        if self.engine:
            self.engine.dispose()
            logger.info("Connexion fermée")