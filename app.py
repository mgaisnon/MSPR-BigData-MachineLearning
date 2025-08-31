import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sys
from pathlib import Path
import logging

# Configuration du logging
logging.getLogger('mysql').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

# Imports des modules
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent))

try:
    from src.data_collection.db_collector import ElectionDBCollector
    from src.data_processing.election_processor import ElectionDataProcessor
    from src.models.election_models import ElectionPredictor
except ImportError as e:
    st.error(f"❌ Erreur d'import: {e}")
    st.info("💡 Vérifiez la structure des dossiers src/")
    st.stop()

# Configuration Streamlit
st.set_page_config(
    page_title="🗳️ Élections Occitanie - Analyse & Prédictions",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_election_data():
    """Charge les données depuis VOTRE base MySQL"""
    try:
        # 🔧 CONNEXION DIRECTE À VOTRE BASE MySQL
        st.info("🔌 Connexion à votre base MySQL...")
        collector = ElectionDBCollector()
        
        # Informations sur votre table
        table_info = collector.get_table_info()
        st.success(f"✅ Connecté à la table '{table_info['table_name']}'")
        
        # Chargement des données
        data = collector.get_election_data()
        
        if data.empty:
            st.error("❌ Aucune donnée trouvée dans votre base pour l'Occitanie")
            st.info("💡 Vérifiez que votre table contient des données pour les départements d'Occitanie")
            return pd.DataFrame()
        
        st.success(f"✅ {len(data)} enregistrements chargés depuis votre base MySQL")
        
        # Debug info
        with st.expander("🔍 Informations sur vos données"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total enregistrements", len(data))
            with col2:
                st.metric("🏛️ Départements", data['departement'].nunique() if 'departement' in data.columns else 0)
            with col3:
                st.metric("🎭 Nuances politiques", data['nuance'].nunique() if 'nuance' in data.columns else 0)
            
            st.write("**Structure de votre table:**")
            st.json(table_info)
            
            st.write("**Aperçu des données:**")
            st.dataframe(data.head(10))
        
        collector.close()
        return data
        
    except Exception as e:
        st.error(f"❌ Erreur de connexion à votre base MySQL: {e}")
        st.info("💡 Vérifications à faire:")
        st.write("1. MySQL est-il démarré ?")
        st.write("2. Vos identifiants dans le .env sont-ils corrects ?")
        st.write("3. Votre utilisateur a-t-il les droits sur la base ?")
        st.write("4. La base contient-elle des données pour l'Occitanie ?")
        
        # Bouton de diagnostic
        if st.button("🔧 Lancer le diagnostic"):
            try:
                from config.config import DB_CONFIG
                st.write("**Configuration actuelle:**")
                st.json({
                    "host": DB_CONFIG.host,
                    "database": DB_CONFIG.database,
                    "user": DB_CONFIG.username,
                    "password_configured": bool(DB_CONFIG.password)
                })
            except Exception as diag_error:
                st.error(f"Erreur diagnostic: {diag_error}")
        
        return pd.DataFrame()

def main():
    """Application principale"""
    st.title("🗳️ Analyse Électorale Occitanie")
    st.markdown("*Analyse et prédictions basées sur votre base MySQL*")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    
    # Chargement des données depuis votre MySQL
    with st.spinner("🔄 Connexion à votre base MySQL..."):
        df_raw = load_election_data()
    
    if df_raw.empty:
        st.stop()
    
    # Menu de navigation
    pages = {
        "📊 Tableau de Bord": "dashboard",
        "🔍 Analyse Exploratoire": "analysis", 
        "📈 Visualisations": "visualizations",
        "🤖 Machine Learning": "ml",
        "🔮 Prédictions": "predictions"
    }
    
    selected_page = st.sidebar.selectbox("Choisissez une section", list(pages.keys()))
    page_key = pages[selected_page]
    
    # Traitement des données
    try:
        processor = ElectionDataProcessor()
        df_processed = processor.preprocess_election_data(df_raw)
        
        if df_processed.empty:
            st.error("❌ Erreur lors du traitement des données")
            return
        
    except Exception as e:
        st.error(f"❌ Erreur traitement données: {e}")
        st.write("Tentative avec données brutes...")
        df_processed = df_raw.copy()
    
    # Affichage des pages
    if page_key == "dashboard":
        show_dashboard(df_processed)
    elif page_key == "analysis":
        show_analysis(df_processed)
    elif page_key == "visualizations":
        show_visualizations(df_processed)
    elif page_key == "ml":
        show_ml_results(df_processed, processor)
    elif page_key == "predictions":
        show_predictions(df_processed, processor)

def show_dashboard(df):
    """Tableau de bord avec vos données MySQL"""
    st.header("📊 Tableau de Bord - Données de votre Base MySQL")
    
    if df.empty:
        st.warning("Aucune donnée à afficher")
        return
    
    # Métriques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("📊 Total enregistrements", f"{total_records:,}")
    
    with col2:
        if 'annee' in df.columns:
            annees = df['annee'].nunique()
            st.metric("📅 Années couvertes", annees)
        else:
            st.metric("📅 Années", "N/A")
    
    with col3:
        if 'departement' in df.columns:
            depts = df['departement'].nunique()
            st.metric("🏛️ Départements", depts)
        else:
            st.metric("🏛️ Départements", "N/A")
    
    with col4:
        if 'nuance' in df.columns:
            nuances = df['nuance'].nunique()
            st.metric("🎭 Nuances politiques", nuances)
        else:
            st.metric("🎭 Nuances", "N/A")
    
    # Aperçu des données
    st.subheader("📋 Aperçu de vos données")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Distribution par année si disponible
    if 'annee' in df.columns:
        st.subheader("📊 Répartition par année")
        year_counts = df['annee'].value_counts().sort_index()
        fig = px.bar(x=year_counts.index, y=year_counts.values, 
                    title="Nombre d'enregistrements par année")
        st.plotly_chart(fig, use_container_width=True)

def show_analysis(df):
    """Analyse exploratoire de vos données"""
    st.header("🔍 Analyse Exploratoire - Vos Données MySQL")
    
    if df.empty:
        st.warning("Aucune donnée à analyser")
        return
    
    # Informations générales
    st.subheader("📊 Informations sur vos données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Colonnes disponibles:**")
        for col in df.columns:
            st.write(f"- {col} ({df[col].dtype})")
    
    with col2:
        st.write("**Statistiques descriptives:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        else:
            st.write("Aucune colonne numérique détectée")
    
    # Analyse par département si disponible
    if 'departement' in df.columns:
        st.subheader("🏛️ Analyse par département")
        dept_counts = df['departement'].value_counts()
        fig = px.bar(x=dept_counts.values, y=dept_counts.index, 
                    orientation='h', title="Enregistrements par département")
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df):
    """Visualisations de vos données"""
    st.header("📈 Visualisations - Données MySQL")
    
    if df.empty:
        st.warning("Aucune donnée à visualiser")
        return
    
    # Sélection des colonnes à visualiser
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 0:
        st.subheader("📊 Visualisations numériques")
        selected_num_col = st.selectbox("Choisissez une colonne numérique", numeric_cols)
        
        if selected_num_col:
            fig = px.histogram(df, x=selected_num_col, title=f"Distribution de {selected_num_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    if len(categorical_cols) > 0:
        st.subheader("🎭 Visualisations catégorielles")
        selected_cat_col = st.selectbox("Choisissez une colonne catégorielle", categorical_cols)
        
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts().head(20)
            fig = px.bar(x=value_counts.values, y=value_counts.index,
                        orientation='h', title=f"Top 20 - {selected_cat_col}")
            st.plotly_chart(fig, use_container_width=True)

def show_ml_results(df_processed, processor):  # ← Ajout des paramètres
    st.header("🤖 Machine Learning")
    st.info("Section ML - En cours de développement pour vos données spécifiques")
    
    # Afficher d'abord les infos sur les données chargées
    if df_processed is not None:
        st.success(f"✅ Données chargées : {len(df_processed)} enregistrements")
        
        # Informations sur les données
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Total enregistrements", len(df_processed))
            if 'famille_politique' in df_processed.columns:
                st.metric("👥 Familles politiques", df_processed['famille_politique'].nunique())
        
        with col2:
            if 'annee' in df_processed.columns:
                years = df_processed['annee'].unique()
                st.metric("📅 Années analysées", f"{min(years)}-{max(years)}")
            if 'departement' in df_processed.columns:
                st.metric("🗺️ Départements", df_processed['departement'].nunique())
    
    # Charger les métadonnées des modèles ML
    try:
        from pathlib import Path
        import joblib
        
        models_dir = Path("models")
        metadata_file = models_dir / "metadata.joblib"
        
        if metadata_file.exists():
            metadata = joblib.load(metadata_file)
            
            st.success("🤖 Modèles ML trouvés et chargés")
            
            # Afficher les performances
            st.subheader("📊 Performances des Modèles")
            
            results = metadata.get('last_results', {})
            
            if results:
                # Créer un DataFrame pour afficher les résultats
                import pandas as pd
                
                perf_data = []
                for model_name, result in results.items():
                    if 'accuracy' in result:
                        perf_data.append({
                            'Modèle': model_name,
                            'Accuracy': f"{result['accuracy']:.1%}",  # Format pourcentage
                            'Données Train': result.get('n_train', 'N/A'),
                            'Données Test': result.get('n_test', 'N/A'),
                            'Features': result.get('n_features', 'N/A')
                        })
                
                if perf_data:
                    df_perf = pd.DataFrame(perf_data)
                    st.dataframe(df_perf, use_container_width=True)
                    
                    # Meilleur modèle
                    best_model = metadata.get('best_model_name')
                    best_score = metadata.get('best_score', 0)
                    
                    st.success(f"🏆 **Meilleur modèle** : {best_model}")
                    st.metric("Accuracy du meilleur modèle", f"{best_score:.1%}")
                    
                    # Graphique de comparaison
                    if len(perf_data) > 1:
                        st.subheader("📈 Comparaison des Modèles")
                        
                        # Préparer les données pour le graphique
                        chart_data = pd.DataFrame({
                            'Modèle': [item['Modèle'] for item in perf_data],
                            'Accuracy': [float(item['Accuracy'].rstrip('%'))/100 for item in perf_data]
                        })
                        
                        import plotly.express as px
                        fig = px.bar(
                            chart_data, 
                            x='Modèle', 
                            y='Accuracy',
                            title='Comparaison des Performances',
                            color='Accuracy',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(yaxis_tickformat='.1%')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Informations supplémentaires
                    st.subheader("ℹ️ Informations Techniques")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("**Configuration des Modèles**")
                        st.write("🌳 **RandomForest**")
                        st.write("• 100 arbres de décision")
                        st.write("• Profondeur max : 10")
                        st.write("• Parallélisation activée")
                        st.write("")
                        st.write("📊 **Logistic Regression**") 
                        st.write("• Régularisation L2")
                        st.write("• Normalisation des données")
                        st.write("• 1000 itérations max")
                    
                    with col2:
                        st.info("**Variables d'Analyse**")
                        st.write("• **Temporelles** : année, décennie")
                        st.write("• **Géographiques** : département, urbain/rural")
                        st.write("• **Électorales** : participation, voix")
                        st.write("• **Dérivées** : influence, tendances")
                        st.write("")
                        st.write("• **Classes** : 6 familles politiques")
                        st.write("• **Période** : 1993-2022 (30 ans)")
            else:
                st.warning("Aucun résultat de performance trouvé")
        else:
            st.warning("⚠️ Aucun modèle ML trouvé")
            st.info("Lancez d'abord : `python main.py --predict`")
            
            # Bouton pour lancer l'entraînement
            if st.button("🚀 Lancer l'entraînement ML"):
                st.info("Commande à exécuter : `python main.py --predict`")
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles : {e}")
        
        # Affichage de fallback avec les données disponibles
        if df_processed is not None:
            st.info("📊 **Analyse des données disponibles**")
            
            # Analyse rapide des données
            if 'famille_politique' in df_processed.columns:
                st.subheader("Distribution des Familles Politiques")
                famille_counts = df_processed['famille_politique'].value_counts()
                
                import plotly.express as px
                fig = px.pie(
                    values=famille_counts.values, 
                    names=famille_counts.index,
                    title="Répartition des Familles Politiques"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.info("Cette section sera complète après l'entraînement des modèles.")

def show_predictions(df_processed, processor):
    st.header("🔮 Prédictions Électorales Réelles")
    st.info("🎯 **Prédictions basées sur vos données MySQL avec vraies nuances politiques**")
    
    # Charger le vrai prédicteur
    try:
        from src.prediction.real_prediction import RealElectionPredictor
        predictor = RealElectionPredictor()
        model_info = predictor.get_model_info()
        
        # Afficher les informations du modèle
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 Modèle", model_info['model_name'])
        with col2:
            st.metric("🎯 Accuracy", f"{model_info['accuracy']:.1%}")
        with col3:
            st.metric("📊 Données", f"{model_info['data_points']:,}")
        with col4:
            st.metric("🔌 MySQL", model_info['mysql_status'])
        
        st.success("✅ Prédicteur chargé avec vos données MySQL")
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")
        st.info("Vérifiez que Laragon est démarré et que le fichier real_predictor.py existe")
        return
    
    # Interface de prédiction
    st.subheader("🎯 Simulateur de Prédiction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⏰ Paramètres Temporels**")
            annee = st.selectbox("Année d'élection", [2025, 2027, 2030, 2032], index=0)
            tour = st.selectbox("Tour", [1, 2], index=0)
            
            st.write("**🗺️ Localisation**")
            # Départements selon vos vraies données
            departements_real = {
                "09 - Ariège": 9,
                "11 - Aude": 11,
                "12 - Aveyron": 12, 
                "30 - Gard": 30,
                "31 - Haute-Garonne": 31,
                "32 - Gers": 32,
                "34 - Hérault": 34,
                "46 - Lot": 46,
                "48 - Lozère": 48,
                "65 - Hautes-Pyrénées": 65,
                "66 - Pyrénées-Orientales": 66,
                "81 - Tarn": 81,
                "82 - Tarn-et-Garonne": 82
            }
            dept_name = st.selectbox("Département", list(departements_real.keys()))
            departement = departements_real[dept_name]
            
            typologie = st.selectbox("Typologie", ["Urbaine", "Rurale"])
            
            # Région historique
            midi_pyrenees_depts = [9, 12, 31, 32, 46, 65, 81, 82]
            if departement in midi_pyrenees_depts:
                region = st.selectbox("Ancienne région", 
                                     ["Midi-Pyrénées"], index=0)
            else:
                region = st.selectbox("Ancienne région", 
                                     ["Languedoc-Roussillon"], index=0)
        
        with col2:
            st.write("**🗳️ Paramètres Électoraux**")
            
            # Suggestions basées sur vos données historiques
            if departement in [31, 34]:  # Toulouse, Montpellier
                default_inscrits = 45000
                default_participation = 68.5
            elif departement in [48, 9]:  # Lozère, Ariège
                default_inscrits = 8000
                default_participation = 72.0
            else:
                default_inscrits = 25000
                default_participation = 65.0
            
            inscrits = st.number_input("Nombre d'inscrits", 
                                     min_value=1000, max_value=100000, 
                                     value=default_inscrits, step=500)
            
            taux_participation = st.slider("Taux de participation (%)", 
                                          35.0, 90.0, default_participation, step=0.5)
            
            # Calculs automatiques
            votants = int(inscrits * taux_participation / 100)
            abstentions = inscrits - votants
            
            st.write("**📊 Calculs Automatiques**")
            st.metric("Votants estimés", f"{votants:,}")
            st.metric("Abstentions", f"{abstentions:,}")
            st.metric("Taux d'abstention", f"{100-taux_participation:.1f}%")
        
        predict_button = st.form_submit_button("🚀 Lancer la Prédiction", 
                                              use_container_width=True)
        
        if predict_button:
            st.markdown("---")
            
            # Préparer les données pour la prédiction
            election_data = {
                'annee': annee,
                'departement': departement,
                'tour': tour,
                'inscrits': inscrits,
                'taux_participation': taux_participation,
                'typologie': typologie,
                'region': region
            }
            
            # Afficher les paramètres (CORRECTION pour éviter l'erreur de sérialisation)
            st.subheader("📋 Paramètres de Prédiction")
            params_df = pd.DataFrame({
                'Paramètre': [
                    'Année', 'Tour', 'Département', 'Typologie', 'Région',
                    'Inscrits', 'Participation', 'Votants'
                ],
                'Valeur': [
                    str(annee),                    # ← Convertir en string
                    str(tour),                     # ← Convertir en string
                    dept_name,                     # ← Déjà string
                    typologie,                     # ← Déjà string
                    region,                        # ← Déjà string
                    f"{inscrits:,}",              # ← Déjà string formaté
                    f"{taux_participation:.1f}%", # ← Déjà string formaté
                    f"{votants:,}"                # ← Déjà string formaté
                ]
            })
            st.dataframe(params_df, use_container_width=True)
            
            # Faire la prédiction RÉELLE
            st.subheader("🎯 Résultat de la Prédiction")
            
            with st.spinner("🤖 Analyse avec votre modèle entraîné..."):
                import time
                time.sleep(1.5)  # Simulation du temps de calcul
                
                # VRAIE PRÉDICTION basée sur vos données
                predictions = predictor.predict_election(election_data)
                
                # Résultat principal
                if predictions and len(predictions) > 0:
                    winner = max(predictions.keys(), key=lambda k: predictions[k])
                    confidence = predictions[winner]
                    
                    # Affichage des résultats
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.success(f"🏆 **Prédiction Gagnante**")
                        st.metric(
                            label="Nuance Politique",
                            value=winner,
                            delta=f"{confidence:.1f}%"
                        )
                        
                        # Niveau de confiance
                        if confidence > 35:
                            st.success("✅ Confiance élevée")
                        elif confidence > 25:
                            st.warning("⚠️ Confiance modérée")
                        else:
                            st.info("📊 Élection serrée")
                    
                    with col2:
                        # Graphique des résultats
                        pred_df = pd.DataFrame(
                            list(predictions.items()),
                            columns=['Nuance Politique', 'Pourcentage']
                        ).sort_values('Pourcentage', ascending=True)
                        
                        import plotly.express as px
                        fig = px.bar(
                            pred_df, 
                            y='Nuance Politique',
                            x='Pourcentage',
                            title="Prédiction basée sur vos données historiques MySQL",
                            color='Pourcentage',
                            color_continuous_scale='RdYlBu_r',
                            text='Pourcentage'
                        )
                        fig.update_traces(
                            texttemplate='%{text:.1f}%', 
                            textposition='outside'
                        )
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            xaxis_title="Pourcentage des votes (%)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau détaillé
                    st.subheader("📊 Détail des Prédictions")
                    
                    # Calcul des voix estimées
                    voix_estimees = {}
                    total_voix = int(votants * 0.95)  # En excluant blancs/nuls
                    
                    for nuance, pourcentage in predictions.items():
                        voix_estimees[nuance] = int(total_voix * pourcentage / 100)
                    
                    detailed_df = pd.DataFrame([
                        {
                            'Nuance Politique': nuance,
                            'Pourcentage': f"{prob:.2f}%",
                            'Voix Estimées': f"{voix_estimees[nuance]:,}",
                            'Rang': idx + 1
                        }
                        for idx, (nuance, prob) in enumerate(
                            sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        )
                    ])
                    
                    st.dataframe(detailed_df, use_container_width=True)
                    
                    # Analyse contextuelle
                    st.subheader("📈 Analyse Contextuelle")
                    
                    analyse = []
                    
                    # Analyse départementale
                    if departement in [31, 34]:
                        analyse.append("🏙️ **Effet urbain** : Grandes métropoles, favorables aux nuances de gauche et centre")
                    elif departement in [9, 48]:
                        analyse.append("🏔️ **Effet montagnard** : Zones rurales avec tradition de droite")
                    elif departement in [30, 66]:
                        analyse.append("⚡ **Zones RN** : Départements avec forte implantation RN historique")
                    
                    # Analyse temporelle
                    if annee >= 2025:
                        analyse.append("⏰ **Projection future** : Basée sur les tendances MySQL 2012-2022")
                    
                    # Analyse participation
                    if taux_participation > 70:
                        analyse.append("📈 **Forte participation** : Tend à modérer les votes protestataires")
                    elif taux_participation < 55:
                        analyse.append("📉 **Faible participation** : Peut renforcer RN et extrêmes")
                    
                    for point in analyse:
                        st.info(point)
                    
                    # Note méthodologique
                    st.info(f"""
                    📝 **Méthodologie**
                    - **Source** : {model_info['data_points']:,} résultats de votre base MySQL ({model_info['years_covered']})
                    - **Nuances analysées** : Basées sur vos vraies données historiques d'Occitanie
                    - **Algorithme** : {model_info['model_name']} (Accuracy: {model_info['accuracy']:.1%})
                    - **Variables** : Temporelles, géographiques, électorales, contextuelles
                    - ⚠️ **Avertissement** : Prédiction indicative basée sur les tendances historiques
                    """)
                
                else:
                    st.error("❌ Aucune prédiction générée")
                    st.info("Vérifiez la connexion MySQL et les données historiques")


if __name__ == "__main__":
    main()