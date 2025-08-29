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
        show_ml_section(df_processed, processor)
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

def show_ml_section(df, processor):
    """Section Machine Learning"""
    st.header("🤖 Machine Learning")
    st.info("Section ML - En cours de développement pour vos données spécifiques")
    
    if df.empty:
        st.warning("Aucune donnée pour le ML")
        return
    
    st.write(f"Données disponibles: {len(df)} enregistrements")
    st.write("Cette section sera adaptée selon la structure de vos données.")

def show_predictions(df, processor):
    """Section Prédictions"""  
    st.header("🔮 Prédictions")
    st.info("Section Prédictions - En cours de développement pour votre base MySQL")
    
    if df.empty:
        st.warning("Aucune donnée pour les prédictions")
        return
    
    st.write(f"Données disponibles: {len(df)} enregistrements")
    st.write("Les prédictions seront adaptées selon vos données spécifiques.")

if __name__ == "__main__":
    main()