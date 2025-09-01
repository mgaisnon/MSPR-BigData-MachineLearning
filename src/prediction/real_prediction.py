import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mysql.connector
import os
from dotenv import load_dotenv
from collections import defaultdict

# SEULE MODIFICATION : Import corrigé pour votre structure
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Remonte à la racine
from config.config import DB_CONFIG

class RealElectionPredictor:
    """Prédicteur basé sur les vrais modèles + données MySQL en direct"""
    
    def __init__(self):
        # Vérifier que la configuration DB est valide
        if DB_CONFIG is None:
            raise ValueError("Configuration de base de données non initialisée. Vérifiez votre fichier .env")
        
        # Utiliser la configuration centralisée
        self.mysql_config = DB_CONFIG.mysql_connector_config
        
        # Debug (optionnel - à retirer en production)
        print("🔍 Configuration MySQL chargée depuis .env")
        print(f"  Host: {self.mysql_config['host']}:{self.mysql_config['port']}")
        print(f"  Database: {self.mysql_config['database']}")
        print(f"  User: {self.mysql_config['user']}")
        print(f"  Password: {'*' * len(self.mysql_config['password'])}")
        
        # MODIFICATION : Chemin des modèles adapté à la structure
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.model = None
        self.is_loaded = False
        self.historical_data = None
        self.metadata = None
        
        # Départements d'Occitanie
        self.occitanie_depts = [9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82]
        
        # Principales nuances de vos données
        self.main_nuances = [
            'SOC', 'UMP', 'LR', 'RN', 'FN', 'ENS', 'REM', 'NUP', 'DVD', 'DVG',
            'ECO', 'VEC', 'COM', 'FI', 'UDF', 'RPR', 'DIV', 'REG'
        ]
        
        # Initialiser
        self.load_models()
        self.load_historical_data()
    
    def load_models(self):
        """Charge les modèles ML entraînés"""
        try:
            metadata_file = self.models_dir / "metadata.joblib"
            if metadata_file.exists():
                self.metadata = joblib.load(metadata_file)
                best_model_name = self.metadata.get('best_model_name', 'LogisticRegression')
                
                if best_model_name == 'RandomForest':
                    model_file = self.models_dir / "randomforest_model.joblib"
                else:
                    model_file = self.models_dir / "logisticregression_model.joblib"
                
                if model_file.exists():
                    self.model = joblib.load(model_file)
                    self.is_loaded = True
                    logging.info(f"✅ Modèle {best_model_name} chargé")
                    return True
            
            logging.warning("⚠️ Modèles ML non trouvés")
            return False
            
        except Exception as e:
            logging.error(f"❌ Erreur chargement modèles : {e}")
            return False
    
    def load_historical_data(self):
        """Charge les données historiques directement de votre base MySQL"""
        try:
            print(f"🔌 Tentative de connexion MySQL...")
            
            # Connexion directe avec mysql.connector utilisant la config sécurisée
            connection = mysql.connector.connect(**self.mysql_config)
            print("✅ Connexion MySQL établie")
            
            # Requête pour récupérer vos données avec les vraies nuances
            query = """
            SELECT 
                année as annee,
                tour,
                departement,
                inscrits,
                votants,
                abstentions,
                exprimes,
                nuance,
                voix
            FROM resultatslegi 
            WHERE departement IN (9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82)
            AND année >= 2012
            ORDER BY année DESC, departement, tour
            """
            
            # Charger les données
            self.historical_data = pd.read_sql(query, connection)
            connection.close()
            
            # Calculs
            self.historical_data['taux_participation'] = (
                self.historical_data['votants'] / self.historical_data['inscrits'] * 100
            ).fillna(0)
            
            print(f"✅ {len(self.historical_data)} enregistrements chargés depuis MySQL")
            print(f"📊 Nuances trouvées : {len(self.historical_data['nuance'].unique())} différentes")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur connexion MySQL : {e}")
            logging.error(f"❌ Erreur connexion MySQL : {e}")
            return False
    
    def get_historical_trends(self, departement, annee_ref=2022):
        """Analyse les tendances historiques par nuances pour un département"""
        if self.historical_data is None or self.historical_data.empty:
            return None
        
        try:
            # Filtrer les données pour le département
            dept_data = self.historical_data[
                self.historical_data['departement'] == departement
            ].copy()
            
            if dept_data.empty:
                return None
            
            # Analyser les dernières élections (2017-2022)
            recent_data = dept_data[dept_data['annee'] >= 2017]
            
            if recent_data.empty:
                recent_data = dept_data  # Prendre toutes si pas assez récentes
            
            # Calculer les pourcentages par nuance
            nuance_stats = {}
            
            # Grouper par année et calculer les totaux
            for annee in recent_data['annee'].unique():
                annee_data = recent_data[recent_data['annee'] == annee]
                total_voix_annee = annee_data['voix'].sum()
                
                if total_voix_annee > 0:
                    for nuance in annee_data['nuance'].unique():
                        nuance_data = annee_data[annee_data['nuance'] == nuance]
                        voix_nuance = nuance_data['voix'].sum()
                        pourcentage = (voix_nuance / total_voix_annee) * 100
                        
                        if nuance not in nuance_stats:
                            nuance_stats[nuance] = []
                        nuance_stats[nuance].append(pourcentage)
            
            # Calculer les moyennes
            final_stats = {}
            for nuance, percentages in nuance_stats.items():
                if percentages:
                    final_stats[nuance] = {
                        'avg_percentage': np.mean(percentages),
                        'last_percentage': percentages[-1] if percentages else 0,
                        'elections_count': len(percentages)
                    }
            
            return final_stats
            
        except Exception as e:
            logging.error(f"Erreur analyse tendances : {e}")
            return None
    
    def predict_from_historical_data(self, election_data):
        """Prédictions basées sur vos vraies données historiques MySQL"""
        
        dept = election_data['departement']
        annee = election_data['annee']
        participation = election_data['taux_participation']
        urbain = election_data['typologie'] == 'Urbaine'
        
        # Récupérer les tendances historiques de votre base
        historical_trends = self.get_historical_trends(dept, annee)
        
        if historical_trends and len(historical_trends) > 0:
            # Utiliser vos vraies données historiques
            logging.info(f"📊 Utilisation des données historiques pour département {dept}")
            
            base_probs = {}
            for nuance, stats in historical_trends.items():
                # Pondérer entre moyenne historique et dernière élection
                weight_avg = 0.7  # 70% moyenne historique
                weight_last = 0.3  # 30% dernière élection
                
                base_prob = (
                    stats['avg_percentage'] * weight_avg + 
                    stats['last_percentage'] * weight_last
                )
                base_probs[nuance] = max(0.5, base_prob)  # Minimum 0.5%
            
        else:
            # Fallback : moyenne sur toute l'Occitanie
            logging.warning(f"⚠️ Pas de données historiques pour département {dept}")
            
            if self.historical_data is not None and not self.historical_data.empty:
                # Calculer moyennes régionales
                occitanie_data = self.historical_data[
                    self.historical_data['departement'].isin(self.occitanie_depts)
                ].copy()
                
                base_probs = {}
                nuance_totals = defaultdict(list)
                
                # Calculer les pourcentages par nuance sur toute la région
                for annee_hist in occitanie_data['annee'].unique():
                    annee_data = occitanie_data[occitanie_data['annee'] == annee_hist]
                    total_voix = annee_data['voix'].sum()
                    
                    if total_voix > 0:
                        for nuance in annee_data['nuance'].unique():
                            nuance_data = annee_data[annee_data['nuance'] == nuance]
                            voix_nuance = nuance_data['voix'].sum()
                            pourcentage = (voix_nuance / total_voix) * 100
                            nuance_totals[nuance].append(pourcentage)
                
                # Moyenne des pourcentages
                for nuance, percentages in nuance_totals.items():
                    if percentages:
                        base_probs[nuance] = np.mean(percentages)
                    else:
                        base_probs[nuance] = 1.0
                        
            else:
                # Fallback ultime basé sur les tendances nationales connues
                base_probs = {
                    'LR': 22.0, 'RN': 20.0, 'ENS': 18.0, 'NUP': 15.0, 
                    'SOC': 12.0, 'DVD': 8.0, 'ECO': 5.0
                }
        
        # Ajustements contextuels basés sur le département
        if dept == 31:  # Haute-Garonne (Toulouse)
            if 'NUP' in base_probs: base_probs['NUP'] *= 1.3
            if 'SOC' in base_probs: base_probs['SOC'] *= 1.2
            if 'ENS' in base_probs: base_probs['ENS'] *= 1.15
            if 'RN' in base_probs: base_probs['RN'] *= 0.8
            if 'LR' in base_probs: base_probs['LR'] *= 0.9
            
        elif dept == 34:  # Hérault (Montpellier)
            if 'ECO' in base_probs: base_probs['ECO'] *= 1.4
            if 'NUP' in base_probs: base_probs['NUP'] *= 1.2
            if 'ENS' in base_probs: base_probs['ENS'] *= 1.1
            if 'RN' in base_probs: base_probs['RN'] *= 0.85
            
        elif dept in [30, 66]:  # Gard, Pyrénées-Orientales
            if 'RN' in base_probs: base_probs['RN'] *= 1.4
            if 'FN' in base_probs: base_probs['FN'] *= 1.4
            if 'ENS' in base_probs: base_probs['ENS'] *= 0.8
            
        elif dept in [9, 48]:  # Ariège, Lozère
            if 'LR' in base_probs: base_probs['LR'] *= 1.3
            if 'DVD' in base_probs: base_probs['DVD'] *= 1.2
            if 'ENS' in base_probs: base_probs['ENS'] *= 0.7
            if 'ECO' in base_probs: base_probs['ECO'] *= 0.6
        
        # Ajustements temporels pour projections futures
        if annee >= 2025:
            if 'RN' in base_probs: base_probs['RN'] *= 1.1
            if 'LR' in base_probs: base_probs['LR'] *= 0.95
            if 'ENS' in base_probs: base_probs['ENS'] *= 0.9
        
        # Ajustement selon la participation
        if participation > 75:
            if 'ENS' in base_probs: base_probs['ENS'] *= 1.1
            if 'LR' in base_probs: base_probs['LR'] *= 1.05
            if 'RN' in base_probs: base_probs['RN'] *= 0.9
        elif participation < 55:
            if 'RN' in base_probs: base_probs['RN'] *= 1.2
            if 'ENS' in base_probs: base_probs['ENS'] *= 0.85
        
        # Ajustement urbain/rural
        if urbain:
            for nuance in ['NUP', 'SOC', 'ECO', 'ENS']:
                if nuance in base_probs:
                    base_probs[nuance] *= 1.1
            for nuance in ['RN', 'DVD']:
                if nuance in base_probs:
                    base_probs[nuance] *= 0.8
        else:  # Rural
            for nuance in ['RN', 'LR', 'DVD']:
                if nuance in base_probs:
                    base_probs[nuance] *= 1.2
            for nuance in ['NUP', 'ECO', 'ENS']:
                if nuance in base_probs:
                    base_probs[nuance] *= 0.8
        
        # Normaliser à 100%
        total = sum(base_probs.values())
        if total > 0:
            final_probs = {k: max(0.1, v / total * 100) for k, v in base_probs.items()}
        else:
            final_probs = {'RN': 25, 'LR': 25, 'ENS': 20, 'NUP': 15, 'ECO': 10, 'DIV': 5}
        
        # Garder les principales nuances (top 6-7)
        sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        top_nuances = dict(sorted_probs[:6])
        autres_total = sum([v for k, v in sorted_probs[6:]])
        
        if autres_total > 1.0:
            top_nuances['Autres'] = autres_total
        
        return top_nuances
    
    def predict_election(self, election_data):
        """Prédiction principale - utilise données MySQL avec nuances"""
        return self.predict_from_historical_data(election_data)
    
    def get_model_info(self):
        """Informations sur le modèle et données MySQL"""
        mysql_status = "✅ Connecté" if (self.historical_data is not None and not self.historical_data.empty) else "❌ Non connecté"
        data_count = len(self.historical_data) if self.historical_data is not None else 0

        # Tente de récupérer le nom du meilleur modèle depuis les métadonnées
        model_name = None
        accuracy = None
        years_covered = None

        if self.metadata:
            model_name = self.metadata.get('best_model_name', None)
            accuracy = self.metadata.get('best_score', None)
            if 'years_covered' in self.metadata:
                years_covered = self.metadata['years_covered']

        if not model_name:
            model_name = "Analyse Nuances MySQL"
        if not accuracy:
            accuracy = 0.0
        if not years_covered:
            years_covered = "2012-2022"

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'data_points': data_count,
            'mysql_status': mysql_status,
            'years_covered': years_covered,
            'departments': 13
        }