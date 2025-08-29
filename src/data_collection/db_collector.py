import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text, inspect
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Correction des imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from config.config import DB_CONFIG, OCCITANIE_CONFIG
except ImportError as e:
    print(f"Erreur import config: {e}")
    raise

logger = logging.getLogger(__name__)

class ElectionDBCollector:
    """Collecteur pour base MySQL existante avec détection automatique"""
    
    def __init__(self):
        self.config = DB_CONFIG
        self.occitanie_depts = OCCITANIE_CONFIG.all_departments
        self.engine = None
        self.mysql_connection = None
        self.table_structure = None
        self.table_name = None
        
        # Connexion et détection automatique
        self._connect_and_detect()
    
    def _connect_and_detect(self):
        """Connexion MySQL et détection automatique de la structure"""
        try:
            logger.info("Connexion à votre base MySQL existante...")
            logger.info(f"   Host: {self.config.host}:{self.config.port}")
            logger.info(f"   Base: {self.config.database}")
            logger.info(f"   Utilisateur: {self.config.username}")
            
            # Test avec mysql.connector
            self.mysql_connection = mysql.connector.connect(**self.config.mysql_connector_config)
            
            if self.mysql_connection.is_connected():
                logger.info("Connexion MySQL réussie !")
                
                # Créer l'engine SQLAlchemy
                self.engine = create_engine(self.config.connection_string, echo=False)
                
                # Détecter la structure de votre base
                self._detect_table_structure()
                
                return True
                
        except Error as mysql_error:
            logger.error(f"Erreur MySQL: {mysql_error}")
            self._diagnose_mysql_error(mysql_error)
            raise
            
        except Exception as e:
            logger.error(f"Erreur connexion: {e}")
            raise
    
    def _diagnose_mysql_error(self, error: Error):
        """Diagnostic détaillé des erreurs MySQL"""
        if error.errno == 1045:  # Access denied
            logger.error("ERREUR D'AUTHENTIFICATION")
            logger.error("   Vérifiez votre fichier .env :")
            logger.error(f"   - DB_USER={self.config.username}")
            logger.error(f"   - DB_PASSWORD={'***' if self.config.password else 'VIDE!'}")
            logger.error("   - Vérifiez que l'utilisateur a les droits sur la base")
            
        elif error.errno == 2003:  # Can't connect  
            logger.error("ERREUR DE CONNEXION")
            logger.error("   - Vérifiez que MySQL est démarré")
            logger.error(f"   - Vérifiez l'adresse {self.config.host}:{self.config.port}")
            logger.error("   - Vérifiez les règles de firewall")
            
        elif error.errno == 1049:  # Unknown database
            logger.error(f"BASE DE DONNÉES INTROUVABLE: {self.config.database}")
            logger.error("   Bases disponibles : lancez SHOW DATABASES; dans MySQL")
            
        elif error.errno == 1146:  # Table doesn't exist
            logger.error("TABLE INTROUVABLE")
            logger.error("   Tables disponibles : lancez SHOW TABLES; dans MySQL")
    
    def _detect_table_structure(self):
        """Détecte automatiquement la structure de votre table d'élections"""
        try:
            logger.info("Détection automatique de la structure...")
            
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            logger.info(f"Tables trouvées: {tables}")
            
            # Je vois que votre table s'appelle 'resultatslegi'
            if 'resultatslegi' in tables:
                found_table = 'resultatslegi'
                logger.info(f"Table d'élections détectée: {found_table}")
            else:
                # Recherche de la table d'élections (patterns possibles)
                election_patterns = [
                    'resultat', 'election', 'vote', 'legi'
                ]
                
                found_table = None
                for pattern in election_patterns:
                    matches = [t for t in tables if pattern.lower() in t.lower()]
                    if matches:
                        found_table = matches[0]
                        logger.info(f"Table d'élections détectée: {found_table}")
                        break
                
                if not found_table:
                    # Prendre la première table si aucune ne correspond
                    if tables:
                        found_table = tables[0]
                        logger.warning(f"Utilisation de la première table: {found_table}")
                    else:
                        raise Exception("Aucune table trouvée dans la base")
            
            self.table_name = found_table
            
            # Analyser la structure de la table
            columns = inspector.get_columns(self.table_name)
            self.table_structure = {col['name']: col['type'] for col in columns}
            
            logger.info(f"Structure de {self.table_name}:")
            for col_name, col_type in self.table_structure.items():
                logger.info(f"   {col_name}: {col_type}")
            
            # Détection des colonnes clés
            self.column_mapping = self._map_columns()
            logger.info(f"Mapping des colonnes: {self.column_mapping}")
            
            # Test de récupération
            with self.engine.begin() as conn:
                test_query = f"SELECT COUNT(*) as count FROM {self.table_name}"
                result = conn.execute(text(test_query))
                total_records = result.fetchone()[0]
                logger.info(f"{total_records} enregistrements dans la table")
                
                # Test avec départements d'Occitanie
                dept_col = self.column_mapping.get('departement', 'departement')
                dept_test = f"SELECT COUNT(*) as count FROM {self.table_name} WHERE {dept_col} IN ('31', '34', '30')"
                result = conn.execute(text(dept_test))
                occitanie_records = result.fetchone()[0]
                logger.info(f"{occitanie_records} enregistrements pour l'Occitanie (échantillon)")
            
        except Exception as e:
            logger.error(f"Erreur détection structure: {e}")
            raise
    
    def _map_columns(self) -> Dict[str, str]:
        """Mapping automatique des colonnes selon différentes conventions"""
        mapping = {}
        columns = list(self.table_structure.keys())
        
        # Patterns de colonnes communes
        patterns = {
            'annee': ['annee', 'year', 'election_year', 'an'],
            'tour': ['tour', 'round', 'ballot'],
            'departement': ['departement', 'dept', 'department', 'code_dept', 'codedept'],
            'nuance': ['nuance', 'parti', 'party', 'political_party', 'couleur'],
            'voix': ['voix', 'votes', 'nb_voix', 'nombre_voix'],
            'inscrits': ['inscrits', 'registered', 'nb_inscrits'],
            'votants': ['votants', 'voters', 'nb_votants'],
            'abstentions': ['abstentions', 'abstention', 'nb_abstentions'],
            'exprimes': ['exprimes', 'valid_votes', 'nb_exprimes']
        }
        
        for key, possible_names in patterns.items():
            found_col = None
            for possible in possible_names:
                matches = [col for col in columns if possible.lower() in col.lower()]
                if matches:
                    found_col = matches[0]
                    break
            
            if found_col:
                mapping[key] = found_col
            else:
                # Fallback sur le nom standard
                mapping[key] = key
        
        return mapping
    
    def get_election_data(self, annees: List[int] = None, tours: List[int] = None) -> pd.DataFrame:
        """Récupère les données depuis votre table MySQL"""
        try:
            if not self.engine or not self.table_name:
                raise Exception("Base de données non connectée ou table non détectée")
            
            # Construction de la requête adaptée à votre structure
            columns_to_select = []
            for std_name, actual_name in self.column_mapping.items():
                if actual_name in self.table_structure:
                    columns_to_select.append(f"{actual_name} as {std_name}")
                else:
                    # Colonne manquante, utiliser une valeur par défaut
                    if std_name in ['inscrits', 'votants', 'abstentions', 'exprimes', 'voix']:
                        columns_to_select.append(f"0 as {std_name}")
                    else:
                        columns_to_select.append(f"'' as {std_name}")
            
            select_clause = ", ".join(columns_to_select)
            
            # Conditions WHERE
            where_conditions = []
            
            # Filtrage par départements d'Occitanie
            dept_col = self.column_mapping['departement']
            if dept_col in self.table_structure:
                dept_list = "','".join(self.occitanie_depts)
                where_conditions.append(f"{dept_col} IN ('{dept_list}')")
            
            # Filtrage par années
            if annees:
                annee_col = self.column_mapping['annee']
                if annee_col in self.table_structure:
                    annee_list = ",".join(map(str, annees))
                    where_conditions.append(f"{annee_col} IN ({annee_list})")
            
            # Filtrage par tours
            if tours:
                tour_col = self.column_mapping['tour']
                if tour_col in self.table_structure:
                    tour_list = ",".join(map(str, tours))
                    where_conditions.append(f"{tour_col} IN ({tour_list})")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
            SELECT {select_clause}
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {self.column_mapping.get('annee', 'annee')} DESC, 
                     {self.column_mapping.get('tour', 'tour')}, 
                     {self.column_mapping.get('departement', 'departement')}
            LIMIT 10000
            """
            
            logger.info("Exécution de la requête sur votre base...")
            logger.debug(f"Query: {query}")
            
            with self.engine.begin() as conn:
                df = pd.read_sql(text(query), conn)
            
            if df.empty:
                logger.warning("Aucune donnée trouvée avec les critères spécifiés")
                logger.info("Vérifiez les données dans votre table pour l'Occitanie")
                return df
            
            # Nettoyage et conversion des types
            df = self._clean_dataframe(df)
            
            logger.info(f"{len(df)} enregistrements chargés depuis votre base MySQL")
            logger.info(f"   Années: {sorted(df['annee'].unique()) if 'annee' in df.columns else 'N/A'}")
            logger.info(f"   Départements: {df['departement'].nunique() if 'departement' in df.columns else 'N/A'}")
            logger.info(f"   Nuances: {df['nuance'].nunique() if 'nuance' in df.columns else 'N/A'}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et standardise le DataFrame"""
        try:
            # Conversion des types numériques
            numeric_cols = ['annee', 'tour', 'inscrits', 'votants', 'abstentions', 'exprimes', 'voix']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
            
            # Nettoyage des colonnes texte
            text_cols = ['departement', 'nuance']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.upper()
            
            # Filtrage final sur les départements d'Occitanie
            if 'departement' in df.columns:
                df = df[df['departement'].isin(self.occitanie_depts)]
            
            # Ajouter un ID si manquant
            if 'id' not in df.columns:
                df['id'] = range(1, len(df) + 1)
                df['id'] = df['id'].astype('int32')
            
            return df
            
        except Exception as e:
            logger.warning(f"Erreur nettoyage données: {e}")
            return df
    
    def get_available_elections(self) -> pd.DataFrame:
        """Liste des élections disponibles dans votre base"""
        try:
            if not self.engine or not self.table_name:
                raise Exception("Base non connectée")
            
            annee_col = self.column_mapping.get('annee', 'annee')
            tour_col = self.column_mapping.get('tour', 'tour')
            
            query = f"""
            SELECT DISTINCT {annee_col} as annee, {tour_col} as tour
            FROM {self.table_name}
            ORDER BY {annee_col} DESC, {tour_col}
            """
            
            with self.engine.begin() as conn:
                df = pd.read_sql(text(query), conn)
            
            logger.info(f"{len(df)} élections trouvées dans votre base")
            return df
            
        except Exception as e:
            logger.error(f"Erreur liste élections: {e}")
            # Fallback
            return pd.DataFrame({
                'annee': [2022, 2022, 2017, 2017],
                'tour': [1, 2, 1, 2]
            })
    
    def get_table_info(self) -> Dict:
        """Informations sur votre table"""
        return {
            'table_name': self.table_name,
            'columns': list(self.table_structure.keys()),
            'column_mapping': self.column_mapping,
            'total_columns': len(self.table_structure),
            'occitanie_departments': self.occitanie_depts
        }
    
    def close(self):
        """Ferme les connexions"""
        try:
            if self.mysql_connection and self.mysql_connection.is_connected():
                self.mysql_connection.close()
                logger.info("Connexion MySQL fermée")
            if self.engine:
                self.engine.dispose()
                logger.info("Engine SQLAlchemy fermé")
        except Exception as e:
            logger.warning(f"Erreur fermeture: {e}")

# Test du collecteur
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("TEST CONNEXION À VOTRE BASE MySQL")
    print("=" * 50)
    
    try:
        collector = ElectionDBCollector()
        
        # Infos sur la table
        info = collector.get_table_info()
        print(f"\nINFORMATIONS SUR VOTRE TABLE:")
        print(f"   Table: {info['table_name']}")
        print(f"   Colonnes: {len(info['columns'])}")
        print(f"   Mapping: {info['column_mapping']}")
        
        # Test données
        print(f"\nTEST RÉCUPÉRATION DONNÉES:")
        data = collector.get_election_data()
        if not data.empty:
            print(f"   {len(data)} enregistrements récupérés")
            print(f"\nAperçu des données:")
            print(data.head())
        else:
            print(f"   Aucune donnée récupérée")
        
        # Test élections
        elections = collector.get_available_elections()
        print(f"\nÉlections disponibles: {len(elections)}")
        print(elections)
        
        collector.close()
        print(f"\nTest terminé avec succès !")
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        print(f"\nVÉRIFICATIONS À FAIRE:")
        print("1. MySQL est-il démarré ?")
        print("2. Le fichier .env est-il correct ?")
        print("3. L'utilisateur a-t-il les droits sur la base ?")
        print("4. La base contient-elle des données ?")