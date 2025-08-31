import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from config.config import API_URLS, OCCITANIE_CONFIG, DATA_CONFIG
import time

logger = logging.getLogger(__name__)

class ElectionDataCollector:
    """Collecteur de données électorales depuis les APIs gouvernementales"""
    
    def __init__(self):
        self.occitanie_depts = OCCITANIE_CONFIG.all_departments
        self.session = requests.Session()
        
    def collect_election_results(self, election_type: str, year: int) -> pd.DataFrame:
        """
        Collecte les résultats électoraux pour un type d'élection et une année
        
        Args:
            election_type: 'presidentielle', 'legislative', 'municipale'
            year: Année de l'élection
        """
        try:
            logger.info(f"Collecte des données {election_type} {year}")
            
            # URL de base pour les données électorales
            base_url = "https://www.data.gouv.fr/fr/datasets/r/"
            
            # Mapping des URLs selon le type d'élection
            urls = {
                'presidentielle_2022': "https://www.data.gouv.fr/fr/datasets/r/eb9102d9-8d5c-4e95-957d-2c5fac8d5b40",
                'presidentielle_2017': "https://www.data.gouv.fr/fr/datasets/r/de4e7b50-3631-47be-b9e1-a07f1f0b3a42",
                'legislative_2022': "https://www.data.gouv.fr/fr/datasets/r/f3dba5fa-c35b-4b26-9b7c-9baaffb25321",
                'legislative_2017': "https://www.data.gouv.fr/fr/datasets/r/58d6d0a9-6e39-4297-aed4-2f28b3d3b302"
            }
            
            url_key = f"{election_type}_{year}"
            if url_key not in urls:
                logger.warning(f"URL non disponible pour {election_type} {year}")
                return self._generate_sample_data(election_type, year)
            
            response = self.session.get(urls[url_key])
            response.raise_for_status()
            
            # Sauvegarde des données brutes
            filename = DATA_CONFIG.raw_data_dir / f"{election_type}_{year}.csv"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Lecture et traitement
            df = pd.read_csv(filename, encoding='utf-8', sep=';', low_memory=False)
            
            # Filtrage pour l'Occitanie
            if 'Code du département' in df.columns:
                df_occitanie = df[df['Code du département'].isin(self.occitanie_depts)]
            else:
                logger.warning("Colonne département non trouvée")
                df_occitanie = df
            
            logger.info(f"Collecté {len(df_occitanie)} enregistrements pour {election_type} {year}")
            return df_occitanie
            
        except Exception as e:
            logger.error(f"Erreur lors de la collecte {election_type} {year}: {e}")
            return self._generate_sample_data(election_type, year)
    
    def _generate_sample_data(self, election_type: str, year: int) -> pd.DataFrame:
        """Génère des données d'exemple pour les tests"""
        logger.info(f"Génération de données d'exemple pour {election_type} {year}")
        
        data = []
        for dept in self.occitanie_depts:
            # Simulation de plusieurs candidats/listes
            candidates = ['MACRON', 'LE PEN', 'MÉLENCHON', 'ZEMMOUR', 'PÉCRESSE']
            if election_type == 'legislative':
                candidates = ['REN', 'RN', 'LFI', 'LR', 'PS', 'EELV']
            
            total_inscrits = np.random.randint(10000, 100000)
            total_votants = int(total_inscrits * np.random.uniform(0.6, 0.85))
            
            for i, candidate in enumerate(candidates):
                if i == 0:  # Premier candidat a plus de voix
                    voix = int(total_votants * np.random.uniform(0.25, 0.35))
                else:
                    voix = int(total_votants * np.random.uniform(0.05, 0.25))
                
                data.append({
                    'id': len(data) + 1,
                    'annee': year,
                    'tour': 1,
                    'departement': dept,
                    'inscrits': total_inscrits,
                    'votants': total_votants,
                    'abstentions': total_inscrits - total_votants,
                    'exprimes': total_votants - int(total_votants * 0.02), 
                    'nuance': candidate,
                    'voix': voix
                })
        
        return pd.DataFrame(data)

class SocioEconomicDataCollector:
    """Collecteur de données socio-économiques"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def collect_insee_data(self, year: int) -> pd.DataFrame:
        """Collecte les données INSEE"""
        try:
            logger.info(f"Collecte des données INSEE {year}")
            
            # URL pour les données de revenus et pauvreté
            url = "https://www.data.gouv.fr/fr/datasets/r/8fa3634a-0c7d-404d-9b24-7e3b7e0b3a42"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Sauvegarde
            filename = DATA_CONFIG.raw_data_dir / f"insee_data_{year}.csv"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv(filename, encoding='utf-8', sep=';', low_memory=False)
            
            # Filtrage pour l'Occitanie
            if 'DEP' in df.columns:
                df_occitanie = df[df['DEP'].isin(OCCITANIE_CONFIG.all_departments)]
            else:
                df_occitanie = df
            
            return df_occitanie
            
        except Exception as e:
            logger.error(f"Erreur collecte INSEE: {e}")
            return self._generate_sample_socio_data(year)
    
    def _generate_sample_socio_data(self, year: int) -> pd.DataFrame:
        """Génère des données socio-économiques d'exemple"""
        import numpy as np
        
        data = []
        for dept in OCCITANIE_CONFIG.all_departments:
            # Générer plusieurs communes par département
            for i in range(np.random.randint(5, 15)):
                data.append({
                    'DEP': dept,
                    'COM': f"{dept}{i+1:03d}",
                    'LIBCOM': f"Commune-{dept}-{i+1}",
                    'taux_pauvrete': np.random.uniform(8.0, 25.0),
                    'revenu_median': np.random.uniform(18000, 35000),
                    'taux_chomage': np.random.uniform(5.0, 15.0),
                    'part_cadres': np.random.uniform(8.0, 25.0),
                    'part_ouvriers': np.random.uniform(15.0, 35.0),
                    'age_moyen': np.random.uniform(35.0, 50.0),
                    'densite_population': np.random.uniform(20.0, 200.0),
                    'annee': year
                })
        
        return pd.DataFrame(data)

class DataCollectorManager:
    """Gestionnaire principal de collecte"""
    
    def __init__(self):
        self.election_collector = ElectionDataCollector()
        self.socio_collector = SocioEconomicDataCollector()
    
    def collect_all_data(self, years: List[int] = None) -> Dict[str, pd.DataFrame]:
        """Collecte toutes les données nécessaires"""
        if years is None:
            years = [2017, 2022]
        
        datasets = {}
        
        for year in years:
            # Données électorales présidentielles
            datasets[f'presidentielle_{year}'] = self.election_collector.collect_election_results('presidentielle', year)
            
            # Données électorales législatives
            datasets[f'legislative_{year}'] = self.election_collector.collect_election_results('legislative', year)
            
            # Données socio-économiques
            datasets[f'socio_economic_{year}'] = self.socio_collector.collect_insee_data(year)
            
            # Pause entre les requêtes
            time.sleep(1)
        
        logger.info(f"Collecté {len(datasets)} datasets")
        return datasets