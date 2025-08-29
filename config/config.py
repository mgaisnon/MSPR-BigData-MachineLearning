import os
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

# Répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent

@dataclass
class DatabaseConfig:
    """Configuration pour la base de données"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '3306'))
    database: str = os.getenv('DB_NAME', 'elections_occitanie')
    username: str = os.getenv('DB_USER', 'root')
    password: str = os.getenv('DB_PASSWORD', '')
    
    @property
    def connection_string(self) -> str:
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class OccitanieConfig:
    """Configuration spécifique à l'Occitanie"""
    # Départements de l'ancienne région Midi-Pyrénées
    midi_pyrenees_depts: List[str] = None
    # Départements de l'ancienne région Languedoc-Roussillon
    languedoc_roussillon_depts: List[str] = None
    
    def __post_init__(self):
        if self.midi_pyrenees_depts is None:
            self.midi_pyrenees_depts = ['09', '12', '31', '32', '46', '65', '81', '82']
        if self.languedoc_roussillon_depts is None:
            self.languedoc_roussillon_depts = ['11', '30', '34', '48', '66']
        
    @property
    def all_departments(self) -> List[str]:
        return self.midi_pyrenees_depts + self.languedoc_roussillon_depts
    
    @property
    def department_names(self) -> Dict[str, str]:
        return {
            '09': 'Ariège', '11': 'Aude', '12': 'Aveyron', '30': 'Gard',
            '31': 'Haute-Garonne', '32': 'Gers', '34': 'Hérault', '46': 'Lot',
            '48': 'Lozère', '65': 'Hautes-Pyrénées', '66': 'Pyrénées-Orientales',
            '81': 'Tarn', '82': 'Tarn-et-Garonne'
        }

@dataclass
class ModelConfig:
    """Configuration pour les modèles ML"""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    target_accuracy: float = 0.75
    models_dir: Path = ROOT_DIR / 'models'
    
    def __post_init__(self):
        self.models_dir.mkdir(exist_ok=True)

@dataclass
class DataConfig:
    """Configuration pour les données"""
    raw_data_dir: Path = ROOT_DIR / 'data' / 'raw'
    processed_data_dir: Path = ROOT_DIR / 'data' / 'processed'
    
    def __post_init__(self):
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class VisualizationConfig:
    """Configuration pour les visualisations"""
    output_dir: Path = ROOT_DIR / 'visualizations'
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)

# Configuration globale
DB_CONFIG = DatabaseConfig()
OCCITANIE_CONFIG = OccitanieConfig()
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
VIZ_CONFIG = VisualizationConfig()

# URLs des APIs gouvernementales
API_URLS = {
    'elections': 'https://www.data.gouv.fr/api/1/datasets/',
    'insee': 'https://api.insee.fr/donnees-locales/V0.1/',
    'geo_api': 'https://geo.api.gouv.fr/',
    'drees': 'https://data.drees.solidarites-sante.gouv.fr/api/records/1.0/'
}