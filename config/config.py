import os
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# Chargement du .env avec gestion d'erreurs
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Fichier .env chargé depuis: {env_path}")
else:
    # Essayer dans le répertoire parent
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Fichier .env chargé depuis: {env_path}")
    else:
        load_dotenv()  # Fallback standard
        print(f"⚠️ Fichier .env non trouvé, utilisation des variables système")

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class DatabaseConfig:
    """Configuration pour la base de données MySQL"""
    
    def __init__(self):
        # Récupération des variables d'environnement
        self.host = os.getenv('MYSQL_HOST')
        self.port = int(os.getenv('MYSQL_PORT', '3306'))
        self.database = os.getenv('MYSQL_DATABASE') 
        self.username = os.getenv('MYSQL_USER')
        self.password = os.getenv('MYSQL_PASSWORD')
        self.charset = os.getenv('DB_CHARSET', 'utf8mb4')
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validation de la configuration"""
        required_vars = {
            'MYSQL_HOST': self.host,
            'MYSQL_DATABASE': self.database,
            'MYSQL_USER': self.username,
            'MYSQL_PASSWORD': self.password
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            print("❌ ERREUR: Variables d'environnement manquantes:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\n💡 Créez un fichier .env avec ces variables:")
            print("MYSQL_HOST=localhost")
            print("MYSQL_DATABASE=bddelections") 
            print("MYSQL_USER=votre_utilisateur")
            print("MYSQL_PASSWORD=votre_mot_de_passe")
            
            # Ne pas lever d'exception, juste avertir
            self._use_defaults()
        else:
            print(f"🔧 Configuration MySQL:")
            print(f"   Host: {self.host}:{self.port}")
            print(f"   Database: {self.database}")
            print(f"   User: {self.username}")
            print(f"   Password: {'✅ configuré' if self.password else '❌ vide'}")
    
    def _use_defaults(self):
        """Utiliser des valeurs par défaut si .env manque"""
        if not self.host:
            self.host = 'localhost'
        if not self.database:
            self.database = 'bddelections'
        if not self.username:
            self.username = 'root'
        if not self.password:
            self.password = 'mysql'
        
        print("⚠️ Utilisation des valeurs par défaut")
    
    @property
    def connection_string(self) -> str:
        """Chaîne de connexion SQLAlchemy"""
        base_params = f"charset={self.charset}&connect_timeout=60&autocommit=true"
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?{base_params}"
    
    @property
    def mysql_connector_config(self) -> dict:
        """Configuration pour mysql.connector"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'password': self.password,
            'charset': self.charset,
            'autocommit': True,
            'raise_on_warnings': False,
            'connection_timeout': 60
        }

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

# Configuration globale - Instantiation sécurisée
try:
    DB_CONFIG = DatabaseConfig()
    OCCITANIE_CONFIG = OccitanieConfig()
    MODEL_CONFIG = ModelConfig()
    DATA_CONFIG = DataConfig()
    VIZ_CONFIG = VisualizationConfig()
    
    print("✅ Configuration chargée avec succès")
    
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation de la configuration: {e}")
    # Créer des configurations par défaut
    DB_CONFIG = None

# URLs des APIs
API_URLS = {
    'elections': 'https://www.data.gouv.fr/api/1/datasets/',
    'insee': 'https://api.insee.fr/donnees-locales/V0.1/',
    'geo_api': 'https://geo.api.gouv.fr/',
    'drees': 'https://data.drees.solidarites-sante.gouv.fr/api/records/1.0/'
}

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 TEST CONFIGURATION")
    print("=" * 60)
    print(f"📁 .env path: {env_path}")
    print(f"📁 .env exists: {env_path.exists()}")
    
    if DB_CONFIG:
        print(f"🔗 Connection OK")
        print(f"📊 Models dir: {MODEL_CONFIG.models_dir}")
        print(f"📁 Data dir: {DATA_CONFIG.raw_data_dir}")
    else:
        print("❌ Configuration DB non disponible")