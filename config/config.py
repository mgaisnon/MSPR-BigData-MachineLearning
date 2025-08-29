import os
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# 🔧 CORRECTION : Chargement du .env avec chemin explicite
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Fichier .env chargé depuis: {env_path}")
else:
    load_dotenv()  # Fallback
    print(f"⚠️ Fichier .env non trouvé à {env_path}, tentative de chargement standard")

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class DatabaseConfig:
    """Configuration pour la base de données MySQL existante"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '3306'))
    database: str = os.getenv('DB_NAME', 'bddelections')
    username: str = os.getenv('DB_USER', 'root')
    password: str = os.getenv('DB_PASSWORD', '')
    charset: str = os.getenv('DB_CHARSET', 'utf8mb4')
    
    def __post_init__(self):
        """Validation de la configuration"""
        if not self.password:
            print("⚠️ ATTENTION: DB_PASSWORD est vide dans le .env")
        print(f"🔧 Configuration MySQL:")
        print(f"   Host: {self.host}:{self.port}")
        print(f"   Database: {self.database}")
        print(f"   User: {self.username}")
        print(f"   Password: {'✅ configuré' if self.password else '❌ vide'}")
    
    @property
    def connection_string(self) -> str:
        """Chaîne de connexion SQLAlchemy"""
        base_params = f"charset={self.charset}&connect_timeout=60&autocommit=true"
        if self.password:
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?{base_params}"
        else:
            return f"mysql+pymysql://{self.username}@{self.host}:{self.port}/{self.database}?{base_params}"
    
    @property
    def mysql_connector_config(self) -> dict:
        """Configuration pour mysql.connector"""
        config = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'charset': self.charset,
            'autocommit': True,
            'raise_on_warnings': False,
            'connection_timeout': 60
        }
        if self.password:
            config['password'] = self.password
        return config

@dataclass
class OccitanieConfig:
    """Configuration Occitanie"""
    midi_pyrenees_depts: List[str] = None
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
    """Configuration pour les modèles Machine Learning"""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    target_accuracy: float = 0.75
    models_dir: Path = ROOT_DIR / 'models'
    
    def __post_init__(self):
        """Créer le répertoire models s'il n'existe pas"""
        self.models_dir.mkdir(exist_ok=True)

@dataclass
class DataConfig:
    """Configuration pour les données"""
    raw_data_dir: Path = ROOT_DIR / 'data' / 'raw'
    processed_data_dir: Path = ROOT_DIR / 'data' / 'processed'
    
    def __post_init__(self):
        """Créer les répertoires data s'ils n'existent pas"""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class VisualizationConfig:
    """Configuration pour les visualisations"""
    output_dir: Path = ROOT_DIR / 'visualizations'
    
    def __post_init__(self):
        """Créer le répertoire visualizations s'il n'existe pas"""
        self.output_dir.mkdir(exist_ok=True)

# 🔧 AJOUT : Configuration globale - TOUTES LES INSTANCES
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

# Test de configuration au chargement
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 TEST CONFIGURATION COMPLÈTE")
    print("=" * 60)
    print(f"📁 .env path: {env_path}")
    print(f"📁 .env exists: {env_path.exists()}")
    print(f"🔗 Connection string: {DB_CONFIG.connection_string[:80]}...")
    print(f"📊 Models dir: {MODEL_CONFIG.models_dir}")
    print(f"📁 Data dir: {DATA_CONFIG.raw_data_dir}")
    print(f"📈 Viz dir: {VIZ_CONFIG.output_dir}")
    print(f"🏛️ Départements Occitanie: {len(OCCITANIE_CONFIG.all_departments)}")