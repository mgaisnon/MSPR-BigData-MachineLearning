import pandas as pd
import logging
import argparse
import sys
from pathlib import Path
import os

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / 'src'))

# Ajouter également le répertoire courant
sys.path.append(str(Path(__file__).parent))

try:
    from src.data_collection.db_collector import ElectionDBCollector
    from src.data_processing.election_processor import ElectionDataProcessor
    from src.models.election_models import ElectionPredictor
    from src.visualization.analyzer import ElectionAnalyzer
    from config.config import DATA_CONFIG, MODEL_CONFIG
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("📁 Vérifiez que tous les dossiers et fichiers sont présents:")
    print("   - config/config.py")
    print("   - src/data_collection/db_collector.py") 
    print("   - src/data_processing/election_processor.py")
    print("   - src/models/election_models.py")
    print("   - src/visualization/analyzer.py")
    sys.exit(1)

# Configuration du logging
def setup_logging():
    """Configure le système de logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configuration des handlers
    handlers = [
        logging.StreamHandler(sys.stdout),  # Sortie console
        logging.FileHandler('election_predictor.log', encoding='utf-8')  # Fichier log
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    # Réduire le niveau de logging pour certains modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

def check_directories():
    """Vérifie et crée les répertoires nécessaires"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'visualizations',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Répertoire vérifié/créé: {directory}")

def safe_format_number(value, format_str=".3f", default="N/A"):
    """Formate un nombre de manière sécurisée"""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value == 'N/A':
            return default
        return f"{value:{format_str}}"
    except (ValueError, TypeError):
        return default

def main():
    """Fonction principale du pipeline"""
    parser = argparse.ArgumentParser(
        description='🗳️ Modèle prédictif élections Occitanie',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py --all          # Pipeline complet
  python main.py --collect      # Collecter les données uniquement
  python main.py --train        # Entraîner les modèles uniquement
  python main.py --analyze      # Créer les visualisations uniquement
  python main.py --predict      # Faire des prédictions uniquement
        """
    )
    
    parser.add_argument('--collect', action='store_true', 
                       help='Collecter et traiter les données')
    parser.add_argument('--train', action='store_true', 
                       help='Entraîner les modèles de machine learning')
    parser.add_argument('--analyze', action='store_true', 
                       help='Créer les analyses et visualisations')
    parser.add_argument('--predict', action='store_true', 
                       help='Faire des prédictions avec le modèle')
    parser.add_argument('--all', action='store_true', 
                       help='Exécuter tout le pipeline complet')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbose pour plus de détails')
    
    args = parser.parse_args()
    
    # Configuration du niveau de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Si aucun argument ou --all, exécuter tout le pipeline
    if not any([args.collect, args.train, args.analyze, args.predict]) or args.all:
        args.collect = args.train = args.analyze = True
        logger.info("🚀 Exécution du pipeline complet")
    
    try:
        logger.info("=" * 60)
        logger.info("🗳️ DÉMARRAGE DU PIPELINE ÉLECTIONS OCCITANIE")
        logger.info("=" * 60)
        logger.info(f"👤 Utilisateur: {os.getenv('USER', os.getenv('USERNAME', 'mgaisnon'))}")
        logger.info(f"📅 Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Vérification des répertoires
        check_directories()
        
        # Variables pour suivre l'état
        processed_df = None
        predictor = None
        
        # 1. COLLECTE ET TRAITEMENT DES DONNÉES
        if args.collect:
            logger.info("=" * 40)
            logger.info("📊 ÉTAPE 1: COLLECTE DES DONNÉES")
            logger.info("=" * 40)
            
            try:
                collector = ElectionDBCollector()
                raw_df = collector.get_election_data()
                
                if raw_df.empty:
                    logger.warning("⚠️ Aucune donnée collectée, utilisation des données d'exemple")
                    # Forcer la création de données d'exemple
                    collector._create_sample_data()
                    raw_df = collector.sample_data
                else:
                    logger.info(f"✅ Collecté {len(raw_df):,} enregistrements")
                
                # Sauvegarde des données brutes
                raw_path = Path('data/raw/election_data_raw.csv')
                raw_df.to_csv(raw_path, index=False, encoding='utf-8')
                logger.info(f"💾 Données brutes sauvegardées: {raw_path}")
                
                # Traitement des données
                logger.info("🔄 Traitement des données...")
                processor = ElectionDataProcessor()
                processed_df = processor.process_election_results(raw_df)
                
                logger.info(f"✅ Données traitées: {processed_df.shape}")
                logger.info(f"   - Départements: {processed_df['departement'].nunique()}")
                logger.info(f"   - Années: {sorted(processed_df['annee'].unique())}")
                logger.info(f"   - Nuances: {processed_df['nuance'].nunique()}")
                
                # Sauvegarde des données traitées
                processed_path = Path('data/processed/election_data_processed.csv')
                processed_df.to_csv(processed_path, index=False, encoding='utf-8')
                logger.info(f"💾 Données traitées sauvegardées: {processed_path}")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de la collecte: {e}")
                logger.info("🔄 Tentative de chargement des données existantes...")
                
                # Essayer de charger des données existantes
                processed_path = Path('data/processed/election_data_processed.csv')
                if processed_path.exists():
                    processed_df = pd.read_csv(processed_path)
                    logger.info(f"✅ Données existantes chargées: {processed_df.shape}")
                else:
                    logger.error("❌ Aucune donnée disponible")
                    return
        
        # Chargement des données pour les autres étapes
        if not args.collect and processed_df is None:
            logger.info("📂 Chargement des données existantes...")
            processed_path = Path('data/processed/election_data_processed.csv')
            
            if processed_path.exists():
                processed_df = pd.read_csv(processed_path)
                logger.info(f"✅ Données chargées: {processed_df.shape}")
            else:
                logger.error("❌ Aucune donnée trouvée. Lancez d'abord avec --collect")
                return
        
        if processed_df is None or processed_df.empty:
            logger.error("❌ Aucune donnée disponible pour continuer")
            return
        
        # 2. ENTRAÎNEMENT DES MODÈLES
        if args.train:
            logger.info("=" * 40)
            logger.info("🤖 ÉTAPE 2: ENTRAÎNEMENT DES MODÈLES")
            logger.info("=" * 40)
            
            try:
                processor = ElectionDataProcessor()
                X, y = processor.prepare_ml_features(processed_df)
                
                if X.empty or y.empty:
                    logger.error("❌ Impossible de préparer les features pour le ML")
                    logger.info("💡 Vérifiez la qualité des données")
                    return
                
                logger.info(f"📊 Features préparées: {X.shape}")
                logger.info(f"🎯 Target préparée: {y.shape}")
                logger.info(f"📋 Colonnes features: {list(X.columns)}")
                
                # Initialisation et entraînement des modèles
                predictor = ElectionPredictor()
                predictor.initialize_models()
                
                logger.info(f"🏭 Entraînement de {len(predictor.models)} modèles...")
                results = predictor.train_models(X, y)
                
                # Affichage des résultats
                print("\n" + "=" * 60)
                print("📈 RÉSULTATS DES MODÈLES")
                print("=" * 60)
                
                successful_models = 0
                for name, result in results.items():
                    if 'error' not in result:
                        accuracy = result.get('accuracy', 0)
                        cv_mean = result.get('cv_mean', 0)
                        cv_std = result.get('cv_std', 0)
                        auc = result.get('auc_score')
                        
                        # ✅ Formatage sécurisé du score AUC
                        auc_str = ""
                        if auc is not None and auc != 'N/A':
                            auc_str = f" | AUC: {safe_format_number(auc)}"
                        
                        accuracy_str = safe_format_number(accuracy)
                        cv_mean_str = safe_format_number(cv_mean)
                        cv_std_str = safe_format_number(cv_std)
                        
                        print(f"🔸 {name:20} | Acc: {accuracy_str} | CV: {cv_mean_str}±{cv_std_str}{auc_str}")
                        successful_models += 1
                    else:
                        error_msg = str(result['error'])
                        if len(error_msg) > 50:
                            error_msg = error_msg[:47] + "..."
                        print(f"❌ {name:20} | Erreur: {error_msg}")
                
                print(f"\n✅ Modèles entraînés avec succès: {successful_models}/{len(results)}")
                
                if predictor.best_model is not None:
                    print(f"🏆 Meilleur modèle: {predictor.best_model_name}")
                    print(f"🎯 Score CV: {safe_format_number(predictor.best_score)}")
                    
                    # Optimisation des hyperparamètres
                    logger.info("⚙️ Optimisation des hyperparamètres...")
                    try:
                        optimization_results = predictor.optimize_best_model(X, y)
                        if optimization_results and 'best_score' in optimization_results:
                            improvement = optimization_results.get('score_improvement', 0)
                            best_score_str = safe_format_number(optimization_results['best_score'])
                            improvement_str = safe_format_number(improvement, "+.3f")
                            
                            print(f"📈 Score après optimisation: {best_score_str}")
                            print(f"📊 Amélioration: {improvement_str}")
                            
                            if optimization_results.get('best_params'):
                                print("🔧 Meilleurs paramètres:")
                                for param, value in optimization_results['best_params'].items():
                                    print(f"   - {param}: {value}")
                        else:
                            logger.info("ℹ️ Optimisation non disponible pour ce modèle")
                    except Exception as e:
                        logger.warning(f"⚠️ Optimisation échouée: {e}")
                    
                    # Importance des features
                    try:
                        importance_df = predictor.get_feature_importance(X.columns.tolist())
                        if not importance_df.empty:
                            print("\n🔍 TOP 10 VARIABLES IMPORTANTES")
                            print("-" * 40)
                            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                                importance_str = safe_format_number(row['importance'])
                                print(f"{i:2d}. {row['feature']:20} | {importance_str}")
                    except Exception as e:
                        logger.warning(f"⚠️ Calcul importance échoué: {e}")
                    
                    # Sauvegarde du modèle
                    try:
                        model_path = 'best_election_model.pkl'
                        predictor.save_model(model_path)
                        logger.info(f"💾 Modèle sauvegardé: models/{model_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Sauvegarde échouée: {e}")
                    
                else:
                    logger.error("❌ Aucun modèle n'a pu être entraîné avec succès")
                    return
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
        
        # 3. ANALYSES ET VISUALISATIONS
        if args.analyze:
            logger.info("=" * 40)
            logger.info("📊 ÉTAPE 3: ANALYSES ET VISUALISATIONS")
            logger.info("=" * 40)
            
            try:
                analyzer = ElectionAnalyzer()
                analyses_results = {}
                
                # Chargement du modèle si pas déjà en mémoire
                if not args.train:
                    try:
                        predictor = ElectionPredictor()
                        predictor.load_model('best_election_model.pkl')
                        logger.info("✅ Modèle chargé pour l'analyse")
                    except Exception as e:
                        logger.warning(f"⚠️ Impossible de charger le modèle: {e}")
                        predictor = None
                
                # Analyse de corrélation
                logger.info("🔗 Création de l'analyse de corrélation...")
                try:
                    corr_analysis = analyzer.create_correlation_analysis(processed_df)
                    if corr_analysis:
                        analyses_results['correlation'] = corr_analysis
                        logger.info(f"✅ Analyse de corrélation créée: {corr_analysis.get('plots_created', [])}")
                except Exception as e:
                    logger.warning(f"⚠️ Analyse corrélation échouée: {e}")
                
                # Analyse géographique
                logger.info("🗺️ Création de l'analyse géographique...")
                try:
                    geo_analysis = analyzer.create_geographic_analysis(processed_df)
                    if geo_analysis:
                        analyses_results['geographic'] = geo_analysis
                        logger.info(f"✅ Analyse géographique créée: {geo_analysis.get('plots_created', [])}")
                except Exception as e:
                    logger.warning(f"⚠️ Analyse géographique échouée: {e}")
                
                # Analyse temporelle
                logger.info("📈 Création de l'analyse temporelle...")
                try:
                    temporal_analysis = analyzer.create_temporal_analysis(processed_df)
                    if temporal_analysis:
                        analyses_results['temporal'] = temporal_analysis
                        logger.info(f"✅ Analyse temporelle créée: {temporal_analysis.get('plots_created', [])}")
                except Exception as e:
                    logger.warning(f"⚠️ Analyse temporelle échouée: {e}")
                
                # Dashboard interactif
                logger.info("🎛️ Création du dashboard interactif...")
                try:
                    interactive_fig = analyzer.create_interactive_dashboard(processed_df)
                    if interactive_fig:
                        analyses_results['interactive'] = {'plots_created': ['interactive_dashboard.html']}
                        logger.info("✅ Dashboard interactif créé")
                except Exception as e:
                    logger.warning(f"⚠️ Dashboard interactif échoué: {e}")
                
                # Analyse de performance si modèle disponible
                if predictor and predictor.best_model is not None:
                    logger.info("🎯 Création de l'analyse de performance...")
                    try:
                        processor = ElectionDataProcessor()
                        X, y = processor.prepare_ml_features(processed_df)
                        if not X.empty and not y.empty:
                            perf_analysis = analyzer.create_performance_dashboard(predictor, X, y)
                            if perf_analysis:
                                analyses_results['performance'] = perf_analysis
                                logger.info("✅ Analyse de performance créée")
                    except Exception as e:
                        logger.warning(f"⚠️ Analyse performance échouée: {e}")
                
                # Génération du rapport HTML
                if analyses_results:
                    logger.info("📋 Génération du rapport final...")
                    try:
                        report_path = analyzer.generate_report(analyses_results)
                        
                        print("\n" + "=" * 60)
                        print("📊 ANALYSES CRÉÉES")
                        print("=" * 60)
                        
                        total_plots = 0
                        for analysis_name, results in analyses_results.items():
                            plots = results.get('plots_created', [])
                            total_plots += len(plots)
                            print(f"🔸 {analysis_name:15} | {len(plots)} visualisation(s)")
                            for plot in plots:
                                print(f"   📄 {plot}")
                        
                        print(f"\n✅ Total visualisations: {total_plots}")
                        print(f"📁 Répertoire: visualizations/")
                        
                        if report_path:
                            print(f"📋 Rapport complet: {report_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Génération rapport échouée: {e}")
                
                else:
                    logger.warning("⚠️ Aucune analyse n'a pu être créée")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'analyse: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 4. PRÉDICTIONS
        if args.predict:
            logger.info("=" * 40)
            logger.info("🔮 ÉTAPE 4: PRÉDICTIONS")
            logger.info("=" * 40)
            
            try:
                # Chargement du modèle
                if predictor is None:
                    predictor = ElectionPredictor()
                    predictor.load_model('best_election_model.pkl')
                
                # Préparation des données pour prédiction
                processor = ElectionDataProcessor()
                X, y = processor.prepare_ml_features(processed_df)
                
                if not X.empty:
                    # Prédiction sur un échantillon
                    sample_size = min(10, len(X))
                    sample_X = X.sample(n=sample_size, random_state=42)
                    
                    predictions, probabilities = predictor.predict(sample_X)
                    
                    print(f"\n🔮 EXEMPLES DE PRÉDICTIONS ({sample_size} échantillons)")
                    print("-" * 60)
                    
                    for i, idx in enumerate(sample_X.index):
                        pred = predictions[i]
                        conf_str = ""
                        
                        if probabilities is not None:
                            confidence = probabilities[i].max()
                            conf_str = f" (confiance: {confidence:.1%})"
                        
                        # Décoder la prédiction si nécessaire
                        if hasattr(predictor, 'label_encoders') and 'winner' in predictor.label_encoders:
                            try:
                                pred_decoded = predictor.label_encoders['winner'].inverse_transform([pred])[0]
                                print(f"🎯 Échantillon {idx:3d}: {pred_decoded}{conf_str}")
                            except:
                                print(f"🎯 Échantillon {idx:3d}: Classe {pred}{conf_str}")
                        else:
                            print(f"🎯 Échantillon {idx:3d}: {pred}{conf_str}")
                    
                    # Statistiques des prédictions
                    if probabilities is not None:
                        avg_confidence = probabilities.max(axis=1).mean()
                        high_conf_count = (probabilities.max(axis=1) > 0.8).sum()
                        
                        print(f"\n📊 Statistiques des prédictions:")
                        print(f"   - Confiance moyenne: {avg_confidence:.1%}")
                        print(f"   - Prédictions haute confiance (>80%): {high_conf_count}/{sample_size}")
                
                else:
                    logger.error("❌ Impossible de préparer les données pour la prédiction")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors des prédictions: {e}")
                logger.info("💡 Assurez-vous qu'un modèle a été entraîné (--train)")
        
        # Résumé final
        logger.info("=" * 60)
        logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        
        # Statistiques finales
        if processed_df is not None:
            print(f"\n📊 RÉSUMÉ FINAL")
            print("-" * 30)
            print(f"📋 Enregistrements traités: {len(processed_df):,}")
            print(f"🏛️ Départements couverts: {processed_df['departement'].nunique()}")
            print(f"📅 Années disponibles: {sorted(processed_df['annee'].unique())}")
            print(f"🎭 Nuances politiques: {processed_df['nuance'].nunique()}")
        
        print(f"\n📁 Fichiers générés:")
        print(f"   - Données: data/processed/")
        if args.train:
            print(f"   - Modèles: models/")
        if args.analyze:
            print(f"   - Visualisations: visualizations/")
        print(f"   - Logs: election_predictor.log")
        
        print(f"\n🚀 Étapes suivantes:")
        print(f"   - Lancez: streamlit run app.py")
        print(f"   - Ouvrez: http://localhost:8501")
        
        logger.info(f"🎉 Pipeline terminé pour l'utilisateur: mgaisnon")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrompu par l'utilisateur")
        print("\n⏹️ Pipeline interrompu.")
        
    except Exception as e:
        logger.error(f"💥 Erreur fatale dans le pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ Erreur fatale: {e}")
        print("📋 Consultez le fichier election_predictor.log pour plus de détails")
        raise

if __name__ == "__main__":
    main()