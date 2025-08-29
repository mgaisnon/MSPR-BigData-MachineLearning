#!/usr/bin/env python3
"""
Script principal pour l'analyse électorale en Occitanie avec MySQL
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Ajouter le répertoire courant au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# Configuration du logging SANS EMOJIS pour Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('election_analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Analyse électorale Occitanie avec MySQL')
    parser.add_argument('--collect', action='store_true', help='Collecter les données depuis MySQL')
    parser.add_argument('--process', action='store_true', help='Traiter les données')
    parser.add_argument('--analyze', action='store_true', help='Analyser les données')
    parser.add_argument('--predict', action='store_true', help='Prédictions ML')
    parser.add_argument('--visualize', action='store_true', help='Générer les visualisations')
    parser.add_argument('--all', action='store_true', help='Exécuter toutes les étapes')
    
    args = parser.parse_args()
    
    # Si aucun argument, afficher l'aide
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    logger.info("=" * 60)
    logger.info("ANALYSE ELECTORALE OCCITANIE - MySQL Edition")
    logger.info("=" * 60)
    logger.info("Utilisateur: mgaisnon")
    logger.info("Date: 2025-08-29 23:14:34")
    
    try:
        # Import des modules nécessaires
        from config.config import DB_CONFIG, OCCITANIE_CONFIG, MODEL_CONFIG, DATA_CONFIG, VIZ_CONFIG
        from src.data_collection.db_collector import ElectionDBCollector
        
        logger.info("Configuration et modules chargés")
        logger.info(f"Base MySQL: {DB_CONFIG.database} sur {DB_CONFIG.host}")
        logger.info(f"Départements Occitanie: {len(OCCITANIE_CONFIG.all_departments)}")
        
        # Étapes du pipeline
        if args.all or args.collect:
            logger.info("\n" + "=" * 40)
            logger.info("ÉTAPE 1: COLLECTE DES DONNÉES MySQL")
            logger.info("=" * 40)
            collect_data()
        
        if args.all or args.process:
            logger.info("\n" + "=" * 40)
            logger.info("ÉTAPE 2: TRAITEMENT DES DONNÉES")
            logger.info("=" * 40)
            process_data()
        
        if args.all or args.analyze:
            logger.info("\n" + "=" * 40)
            logger.info("ÉTAPE 3: ANALYSE DES DONNÉES")
            logger.info("=" * 40)
            analyze_data()
        
        if args.all or args.predict:
            logger.info("\n" + "=" * 40)
            logger.info("ÉTAPE 4: MODÈLES PRÉDICTIFS")
            logger.info("=" * 40)
            predict_data()
        
        if args.all or args.visualize:
            logger.info("\n" + "=" * 40)
            logger.info("ÉTAPE 5: VISUALISATIONS")
            logger.info("=" * 40)
            visualize_data()
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE TERMINÉ AVEC SUCCÈS !")
        logger.info("=" * 60)
        logger.info("Vous pouvez maintenant lancer: streamlit run app.py")
        
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        logger.error("Vérifiez que tous les dossiers et fichiers sont présents:")
        logger.error("   - config/config.py")
        logger.error("   - src/data_collection/db_collector.py")
        logger.error("   - src/data_processing/election_processor.py")
        logger.error("   - src/models/election_models.py")
        logger.error("   - src/visualization/analyzer.py")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def collect_data():
    """Collecte des données depuis MySQL"""
    try:
        from src.data_collection.db_collector import ElectionDBCollector
        from config.config import OCCITANIE_CONFIG, DATA_CONFIG
        
        logger.info("Connexion à la base MySQL...")
        collector = ElectionDBCollector()
        
        # Informations sur la base
        table_info = collector.get_table_info()
        logger.info(f"Table utilisée: {table_info['table_name']}")
        logger.info(f"Colonnes détectées: {len(table_info['columns'])}")
        logger.info(f"Mapping: {table_info['column_mapping']}")
        
        # Collecte des données
        logger.info("Récupération des données d'Occitanie...")
        data = collector.get_election_data()
        
        if data.empty:
            logger.warning("Aucune donnée récupérée pour l'Occitanie")
            logger.info("Vérifiez que votre base contient des données pour les départements:")
            logger.info(f"   {', '.join(OCCITANIE_CONFIG.all_departments)}")
        else:
            logger.info(f"{len(data)} enregistrements récupérés")
            logger.info(f"   Années: {sorted(data['annee'].unique()) if 'annee' in data.columns else 'N/A'}")
            logger.info(f"   Départements: {data['departement'].nunique() if 'departement' in data.columns else 'N/A'}")
            logger.info(f"   Nuances: {data['nuance'].nunique() if 'nuance' in data.columns else 'N/A'}")
            
            # Sauvegarde
            output_file = DATA_CONFIG.raw_data_dir / 'elections_occitanie_mysql.csv'
            data.to_csv(output_file, index=False)
            logger.info(f"Données sauvegardées: {output_file}")
        
        # Élections disponibles
        elections = collector.get_available_elections()
        logger.info(f"Élections disponibles: {len(elections)}")
        if not elections.empty:
            for _, row in elections.iterrows():
                logger.info(f"   {row.get('annee', 'N/A')} - Tour {row.get('tour', 'N/A')}")
        
        collector.close()
        
    except Exception as e:
        logger.error(f"Erreur collecte données: {e}")
        raise

def process_data():
    """Traitement des données"""
    try:
        from config.config import DATA_CONFIG
        
        logger.info("Traitement des données...")
        
        # Vérifier si le processeur existe
        try:
            from src.data_processing.election_processor import ElectionDataProcessor
            processor = ElectionDataProcessor()
            logger.info("Processeur de données initialisé")
        except ImportError:
            logger.warning("Module election_processor non trouvé, traitement basique")
            return
        
        # Charger les données brutes
        raw_file = DATA_CONFIG.raw_data_dir / 'elections_occitanie_mysql.csv'
        
        if raw_file.exists():
            data = pd.read_csv(raw_file)
            logger.info(f"Données chargées: {len(data)} enregistrements")
            
            # Traitement
            processed_data = processor.preprocess_election_data(data)
            
            if not processed_data.empty:
                processed_file = DATA_CONFIG.processed_data_dir / 'elections_processed.csv'
                processed_data.to_csv(processed_file, index=False)
                logger.info(f"Données traitées sauvegardées: {processed_file}")
            else:
                logger.warning("Aucune donnée traitée générée")
        else:
            logger.warning(f"Fichier de données brutes non trouvé: {raw_file}")
            logger.info("Lancez d'abord: python main.py --collect")
            
    except Exception as e:
        logger.error(f"Erreur traitement: {e}")

def analyze_data():
    """Analyse des données"""
    try:
        # 🔧 CORRECTION: Import de OCCITANIE_CONFIG dans la fonction
        from config.config import DATA_CONFIG, OCCITANIE_CONFIG
        
        logger.info("Analyse des données...")
        
        # Charger les données traitées
        processed_file = DATA_CONFIG.processed_data_dir / 'elections_processed.csv'
        
        if processed_file.exists():
            data = pd.read_csv(processed_file)
            logger.info(f"Analyse sur {len(data)} enregistrements")
            
            # Analyses basiques
            if 'departement' in data.columns:
                dept_stats = data['departement'].value_counts()
                logger.info("Top 5 départements par enregistrements:")
                for dept, count in dept_stats.head().items():
                    dept_name = OCCITANIE_CONFIG.department_names.get(str(dept), f"Dept {dept}")
                    logger.info(f"   {dept} ({dept_name}): {count}")
            
            if 'nuance' in data.columns:
                nuance_stats = data['nuance'].value_counts()
                logger.info("Top 5 nuances par enregistrements:")
                for nuance, count in nuance_stats.head().items():
                    logger.info(f"   {nuance}: {count}")
            
            # Nouvelles analyses sur données enrichies
            if 'famille_politique' in data.columns:
                famille_stats = data['famille_politique'].value_counts()
                logger.info("Top 5 familles politiques:")
                for famille, count in famille_stats.head().items():
                    logger.info(f"   {famille}: {count}")
            
            if 'taux_participation' in data.columns:
                avg_participation = data['taux_participation'].mean()
                logger.info(f"Taux de participation moyen: {avg_participation:.2f}%")
            
            if 'typologie' in data.columns:
                typo_stats = data['typologie'].value_counts()
                logger.info("Répartition urbain/rural:")
                for typo, count in typo_stats.items():
                    logger.info(f"   {typo}: {count}")
            
            # Analyse par ancienne région
            if 'ancien_midi_pyrenees' in data.columns:
                midi_count = data['ancien_midi_pyrenees'].sum()
                languedoc_count = len(data) - midi_count
                logger.info("Répartition par ancienne région:")
                logger.info(f"   Ancien Midi-Pyrénées: {midi_count}")
                logger.info(f"   Ancien Languedoc-Roussillon: {languedoc_count}")
            
            # Analyse temporelle
            if 'annee' in data.columns:
                annees_stats = data['annee'].value_counts().sort_index()
                logger.info("Répartition par année:")
                for annee, count in annees_stats.items():
                    logger.info(f"   {annee}: {count} enregistrements")
            
            logger.info("Analyse terminée avec succès")
            
        else:
            logger.warning(f"Fichier de données traitées non trouvé: {processed_file}")
            logger.info("Lancez d'abord: python main.py --process")
            
    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Dans la fonction predict_data(), remplacez par :
def predict_data():
    """Modèles prédictifs"""
    try:
        logger.info("Modèles prédictifs...")
        
        # Import du module ML
        try:
            from src.models.election_models import train_election_models
            
            predictor, results = train_election_models()
            
            if predictor and "error" not in results:
                logger.info("Entraînement ML réussi !")
                for model_name, result in results.items():
                    if 'accuracy' in result:
                        logger.info(f"  {model_name}: Accuracy = {result['accuracy']:.4f}")
                        logger.info(f"    Données train: {result.get('n_train', 'N/A')}")
                        logger.info(f"    Données test: {result.get('n_test', 'N/A')}")
                        logger.info(f"    Features: {result.get('n_features', 'N/A')}")
                
                logger.info(f"Meilleur modèle: {predictor.best_model_name}")
                logger.info("Modèles sauvegardés dans ./models/")
            else:
                logger.warning(f"Erreur entraînement ML: {results}")
                
        except ImportError as e:
            logger.error(f"Module ML non trouvé: {e}")
            logger.info("Créez d'abord src/models/election_models.py")
        
    except Exception as e:
        logger.error(f"Erreur prédictions: {e}")

def visualize_data():
    """Visualisations - FONCTION AJOUTÉE"""
    try:
        logger.info("Génération des visualisations...")
        
        # Import du module de visualisation
        try:
            from config.config import DATA_CONFIG
            
            # Vérifier si on a des données traitées
            processed_file = DATA_CONFIG.processed_data_dir / 'elections_processed.csv'
            
            if processed_file.exists():
                data = pd.read_csv(processed_file)
                logger.info(f"Données pour visualisation: {len(data)} enregistrements")
                
                # Statistiques rapides pour les logs
                if 'famille_politique' in data.columns:
                    top_families = data['famille_politique'].value_counts().head(3)
                    logger.info("Top 3 familles politiques pour viz:")
                    for famille, count in top_families.items():
                        logger.info(f"  {famille}: {count}")
                
                if 'taux_participation' in data.columns:
                    avg_participation = data['taux_participation'].mean()
                    logger.info(f"Participation moyenne pour viz: {avg_participation:.2f}%")
                
                logger.info("Données prêtes pour Streamlit")
                logger.info("Visualisations disponibles dans l'interface web")
                
            else:
                logger.warning("Pas de données traitées pour les visualisations")
                logger.info("Lancez d'abord: python main.py --process")
                
        except ImportError as e:
            logger.error(f"Module visualisation: {e}")
            logger.info("Module visualisation simulé")
        
        logger.info("Étape visualisations terminée")
        
    except Exception as e:
        logger.error(f"Erreur visualisations: {e}")

if __name__ == "__main__":
    main()