import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import folium
import os
import logging
from config.config import VIZ_CONFIG, OCCITANIE_CONFIG
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class ElectionAnalyzer:
    """Classe pour les analyses et visualisations électorales"""
    
    def __init__(self):
        self.output_dir = VIZ_CONFIG.output_dir
        self.dept_names = OCCITANIE_CONFIG.department_names
        
        # Configuration des graphiques
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configuration Plotly
        self.plotly_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    
    def create_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Crée l'analyse de corrélation complète"""
        try:
            # Sélection des variables numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                logger.warning("Pas assez de variables numériques pour la corrélation")
                return {}
            
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap matplotlib
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f', square=True)
            plt.title('Matrice de corrélation - Variables électorales')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Heatmap interactive Plotly
            fig = px.imshow(corr_matrix, 
                          title="Matrice de corrélation interactive",
                          color_continuous_scale='RdBu_r',
                          aspect="auto")
            fig.write_html(self.output_dir / 'correlation_interactive.html')
            
            # Analyse des corrélations les plus fortes
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            top_correlations = corr_df.reindex(
                corr_df['correlation'].abs().sort_values(ascending=False).index
            ).head(10)
            
            logger.info("Analyse de corrélation terminée")
            
            return {
                'correlation_matrix': corr_matrix,
                'top_correlations': top_correlations,
                'plots_created': ['correlation_matrix.png', 'correlation_interactive.html']
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse corrélation: {e}")
            return {}
    
    def create_geographic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Crée les analyses géographiques"""
        try:
            if 'departement' not in df.columns:
                logger.warning("Colonne département manquante")
                return {}
            
            # Analyse par département
            dept_stats = df.groupby('departement').agg({
                'taux_participation': 'mean',
                'part_voix': 'mean',
                'voix': 'sum',
                'inscrits': 'sum'
            }).reset_index()
            
            dept_stats['nom_departement'] = dept_stats['departement'].map(self.dept_names)
            
            # Graphiques par département
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Participation par département', 
                              'Voix totales par département',
                              'Part moyenne des voix', 
                              'Inscrits par département'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Participation
            fig.add_trace(
                go.Bar(x=dept_stats['departement'], 
                      y=dept_stats['taux_participation'],
                      name='Taux participation'),
                row=1, col=1
            )
            
            # Voix totales
            fig.add_trace(
                go.Bar(x=dept_stats['departement'], 
                      y=dept_stats['voix'],
                      name='Voix totales'),
                row=1, col=2
            )
            
            # Part des voix
            fig.add_trace(
                go.Bar(x=dept_stats['departement'], 
                      y=dept_stats['part_voix'],
                      name='Part moyenne voix'),
                row=2, col=1
            )
            
            # Inscrits
            fig.add_trace(
                go.Bar(x=dept_stats['departement'], 
                      y=dept_stats['inscrits'],
                      name='Inscrits'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False,
                            title_text="Analyse géographique - Occitanie")
            fig.write_html(self.output_dir / 'geographic_analysis.html')
            
            # Carte de France (simulation - nécessiterait vraies coordonnées)
            self._create_department_map(dept_stats)
            
            logger.info("Analyse géographique terminée")
            
            return {
                'department_stats': dept_stats,
                'plots_created': ['geographic_analysis.html', 'occitanie_map.html']
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse géographique: {e}")
            return {}
    
    def _create_department_map(self, dept_stats: pd.DataFrame):
        """Crée une carte des départements (simulation)"""
        try:
            # Coordonnées approximatives des départements d'Occitanie
            dept_coords = {
                '09': [42.96, 1.60], '11': [43.22, 2.35], '12': [44.35, 2.57],
                '30': [43.84, 4.36], '31': [43.60, 1.44], '32': [43.65, 0.59],
                '34': [43.61, 3.88], '46': [44.45, 1.44], '48': [44.52, 3.50],
                '65': [43.23, 0.08], '66': [42.70, 2.90], '81': [43.93, 2.14],
                '82': [44.02, 1.35]
            }
            
            # Créer la carte centrée sur l'Occitanie
            center_lat = 43.5
            center_lon = 2.0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
            
            # Ajouter les départements
            for _, row in dept_stats.iterrows():
                dept = row['departement']
                if dept in dept_coords:
                    coords = dept_coords[dept]
                    
                    # Taille du cercle basée sur la participation
                    radius = row['taux_participation'] * 50000
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=radius/1000,  # Ajustement échelle
                        popup=f"{dept} - {self.dept_names.get(dept, dept)}<br>"
                              f"Participation: {row['taux_participation']:.1%}<br>"
                              f"Voix: {row['voix']:,.0f}",
                        color='blue',
                        fill=True,
                        weight=2
                    ).add_to(m)
            
            # Sauvegarder
            m.save(str(self.output_dir / 'occitanie_map.html'))
            
        except Exception as e:
            logger.error(f"Erreur création carte: {e}")
    
    def create_temporal_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse l'évolution temporelle"""
        try:
            if 'annee' not in df.columns:
                logger.warning("Colonne année manquante")
                return {}
            
            # Évolution globale par année
            temporal_stats = df.groupby('annee').agg({
                'taux_participation': 'mean',
                'voix': 'sum',
                'inscrits': 'sum',
                'abstentions': 'sum'
            }).reset_index()
            
            # Graphique d'évolution
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Évolution participation', 
                              'Évolution voix totales',
                              'Évolution inscrits', 
                              'Évolution abstentions')
            )
            
            fig.add_trace(
                go.Scatter(x=temporal_stats['annee'], 
                          y=temporal_stats['taux_participation'],
                          mode='lines+markers', 
                          name='Participation'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=temporal_stats['annee'], 
                          y=temporal_stats['voix'],
                          mode='lines+markers', 
                          name='Voix'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=temporal_stats['annee'], 
                          y=temporal_stats['inscrits'],
                          mode='lines+markers', 
                          name='Inscrits'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=temporal_stats['annee'], 
                          y=temporal_stats['abstentions'],
                          mode='lines+markers', 
                          name='Abstentions'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False,
                            title_text="Évolution temporelle - Occitanie")
            fig.write_html(self.output_dir / 'temporal_analysis.html')
            
            # Analyse des tendances par nuance
            if 'nuance' in df.columns:
                self._create_nuance_evolution(df)
            
            logger.info("Analyse temporelle terminée")
            
            return {
                'temporal_stats': temporal_stats,
                'plots_created': ['temporal_analysis.html', 'nuance_evolution.html']
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse temporelle: {e}")
            return {}
    
    def _create_nuance_evolution(self, df: pd.DataFrame):
        """Évolution des nuances politiques"""
        try:
            # Top nuances
            top_nuances = df['nuance'].value_counts().head(8).index.tolist()
            
            nuance_evolution = df[df['nuance'].isin(top_nuances)].groupby(['annee', 'nuance']).agg({
                'voix': 'sum',
                'part_voix': 'mean'
            }).reset_index()
            
            fig = px.line(nuance_evolution, 
                         x='annee', 
                         y='voix', 
                         color='nuance',
                         title="Évolution des principales nuances politiques")
            
            fig.write_html(self.output_dir / 'nuance_evolution.html')
            
        except Exception as e:
            logger.error(f"Erreur évolution nuances: {e}")
    
    def create_performance_dashboard(self, predictor, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Dashboard de performance du modèle"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_curve, precision_recall_curve
            
            # Split pour l'évaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Prédictions
            predictions, probabilities = predictor.predict(X_test)
            
            # Dashboard avec subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Matrice de confusion', 
                              'Distribution des probabilités',
                              'Importance des variables', 
                              'Courbe ROC'),
                specs=[[{"type": "heatmap"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # 1. Matrice de confusion
            cm = confusion_matrix(y_test, predictions)
            fig.add_trace(
                go.Heatmap(z=cm, 
                          colorscale='Blues',
                          showscale=False),
                row=1, col=1
            )
            
            # 2. Distribution des probabilités
            if probabilities is not None and probabilities.shape[1] == 2:
                fig.add_trace(
                    go.Histogram(x=probabilities[:, 1], 
                               nbinsx=20,
                               name='Probabilités classe 1'),
                    row=1, col=2
                )
            
            # 3. Importance des features
            if predictor.feature_importance is not None:
                top_features = predictor.feature_importance.head(10)
                fig.add_trace(
                    go.Bar(x=top_features['importance'],
                          y=top_features['feature'],
                          orientation='h',
                          name='Importance'),
                    row=2, col=1
                )
            
            # 4. Courbe ROC
            if probabilities is not None and len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr,
                             name=f'ROC (AUC = {roc_auc:.2f})'),
                    row=2, col=2
                )
                
                # Ligne de référence
                fig.add_trace(
                    go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             line=dict(dash='dash'),
                             name='Aléatoire'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, 
                            title_text="Dashboard Performance Modèle")
            fig.write_html(self.output_dir / 'model_performance.html')
            
            # Métriques détaillées
            evaluation = predictor.evaluate_model(X_test, y_test)
            
            logger.info("Dashboard performance créé")
            
            return {
                'evaluation': evaluation,
                'plots_created': ['model_performance.html'],
                'confusion_matrix': cm
            }
            
        except Exception as e:
            logger.error(f"Erreur dashboard performance: {e}")
            return {}
    
    def create_interactive_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Dashboard interactif complet"""
        try:
            # Vérifier les colonnes nécessaires
            required_cols = ['departement']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if not available_cols:
                logger.warning("Colonnes nécessaires manquantes pour le dashboard")
                return go.Figure()
            
            # Dashboard principal
            if 'taux_participation' in df.columns and 'part_voix' in df.columns:
                fig = px.scatter(
                    df,
                    x='taux_participation',
                    y='part_voix',
                    color='departement',
                    size='voix' if 'voix' in df.columns else None,
                    hover_data=['nuance'] if 'nuance' in df.columns else None,
                    title='Dashboard Interactif - Relations entre variables'
                )
            else:
                # Dashboard basique
                agg_data = df.groupby('departement').size().reset_index(name='count')
                fig = px.bar(agg_data, 
                           x='departement', 
                           y='count',
                           title='Nombre d\'enregistrements par département')
            
            fig.write_html(self.output_dir / 'interactive_dashboard.html')
            
            logger.info("Dashboard interactif créé")
            return fig
            
        except Exception as e:
            logger.error(f"Erreur dashboard interactif: {e}")
            return go.Figure()
    
    def create_comparison_analysis(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                 labels: List[str] = None) -> Dict[str, Any]:
        """Compare deux datasets"""
        try:
            if labels is None:
                labels = ['Dataset 1', 'Dataset 2']
            
            # Statistiques comparatives
            comparison_stats = pd.DataFrame({
                labels[0]: df1.describe().round(2),
                labels[1]: df2.describe().round(2)
            })
            
            # Graphique de comparaison
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=labels)
            
            # Histogrammes des principales variables
            common_cols = set(df1.columns) & set(df2.columns)
            numeric_cols = [col for col in common_cols 
                           if df1[col].dtype in [np.float64, np.int64]]
            
            if numeric_cols:
                col_to_plot = numeric_cols[0]
                
                fig.add_trace(
                    go.Histogram(x=df1[col_to_plot], name=labels[0]),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=df2[col_to_plot], name=labels[1]),
                    row=1, col=2
                )
            
            fig.update_layout(title_text="Analyse comparative")
            fig.write_html(self.output_dir / 'comparison_analysis.html')
            
            logger.info("Analyse comparative créée")
            
            return {
                'comparison_stats': comparison_stats,
                'common_columns': list(common_cols),
                'plots_created': ['comparison_analysis.html']
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse comparative: {e}")
            return {}
    
    def generate_report(self, analyses: Dict[str, Any]) -> str:
        """Génère un rapport HTML complet"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Rapport d'Analyse Électorale - Occitanie</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                             background-color: #e9ecef; border-radius: 5px; }}
                    .plots {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Rapport d'Analyse Électorale - Région Occitanie</h1>
                <div class="summary">
                    <h2>Résumé Exécutif</h2>
                    <p>Analyse des données électorales pour les 13 départements de la région Occitanie.</p>
                </div>
                
                <h2>Métriques Clés</h2>
            """
            
            # Ajouter les métriques si disponibles
            for analysis_name, results in analyses.items():
                if isinstance(results, dict) and 'plots_created' in results:
                    html_content += f"""
                    <div class="metric">
                        <h3>{analysis_name.replace('_', ' ').title()}</h3>
                        <p>Graphiques générés: {len(results['plots_created'])}</p>
                    </div>
                    """
            
            html_content += """
                <h2>Visualisations</h2>
                <div class="plots">
                    <p>Les graphiques interactifs sont disponibles dans les fichiers HTML générés.</p>
                </div>
            </body>
            </html>
            """
            
            # Sauvegarder le rapport
            report_path = self.output_dir / 'rapport_complet.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Rapport généré: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")
            return ""