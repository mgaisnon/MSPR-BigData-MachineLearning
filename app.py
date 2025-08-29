import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_collection.db_collector import ElectionDBCollector
from src.data_processing.election_processor import ElectionDataProcessor
from src.models.election_models import ElectionPredictor
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur Élections Occitanie",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Supprimer les warnings
logging.getLogger().setLevel(logging.ERROR)

@st.cache_data
def load_election_data():
    """Charge les données électorales depuis la BDD"""
    try:
        collector = ElectionDBCollector()
        return collector.get_election_data()
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return pd.DataFrame()

@st.cache_data
def get_available_elections():
    """Récupère les élections disponibles"""
    try:
        collector = ElectionDBCollector()
        return collector.get_available_elections()
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Charge le modèle entraîné"""
    try:
        predictor = ElectionPredictor()
        predictor.load_model('best_election_model.pkl')
        return predictor
    except Exception as e:
        st.warning(f"Modèle non disponible: {e}")
        return None

def main():
    # En-tête avec style
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🗳️ Analyseur Élections Législatives - Occitanie
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            MSPR - TPRE813 : Piloter l'informatique décisionnel d'un SI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choisir une section",
        [
            "🏠 Accueil", 
            "📊 Données Brutes", 
            "🏛️ Analyses par Département", 
            "📈 Évolution Temporelle", 
            "🤖 Prédictions", 
            "🔄 Comparaisons"
        ]
    )
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        df = load_election_data()
        processor = ElectionDataProcessor()
    
    if df.empty:
        st.error("❌ Impossible de charger les données. Vérifiez la connexion à la base de données.")
        st.info("💡 Lancez d'abord `python main.py --collect` pour générer des données d'exemple.")
        return
    
    # Traitement des données
    df_processed = processor.process_election_results(df)
    
    # Navigation
    if page == "🏠 Accueil":
        show_accueil(df, df_processed)
    elif page == "📊 Données Brutes":
        show_donnees_brutes(df, df_processed, processor)
    elif page == "🏛️ Analyses par Département":
        show_analyses_departement(df_processed, processor)
    elif page == "📈 Évolution Temporelle":
        show_evolution_temporelle(df_processed, processor)
    elif page == "🤖 Prédictions":
        show_predictions(df_processed, processor)
    elif page == "🔄 Comparaisons":
        show_comparaisons(df_processed, processor)

def show_accueil(df, df_processed):
    """Page d'accueil avec overview du projet"""
    st.header("🏠 Présentation du Projet")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total enregistrements", f"{len(df):,}")
    with col2:
        st.metric("🏛️ Départements couverts", df['departement'].nunique())
    with col3:
        st.metric("📅 Années disponibles", df['annee'].nunique())
    with col4:
        st.metric("🎭 Nuances politiques", df['nuance'].nunique())
    
    # Informations sur l'équipe
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👥 Équipe de développement")
        st.markdown("""
        - **Paul CARION**
        - **Yassin FARASSI**
        - **Mathieu GAISNON**
        - **Julie MONTOUX**
        """)
        
        st.subheader("🎯 Objectifs du projet")
        st.markdown("""
        - Analyser les comportements électoraux en Occitanie
        - Créer des modèles prédictifs performants
        - Fournir des outils d'aide à la décision
        - Visualiser les tendances politiques régionales
        """)
    
    with col2:
        st.subheader("🗺️ Périmètre géographique")
        st.markdown("""
        **Région Occitanie (13 départements)**
        
        *Ancien Midi-Pyrénées:*
        - 09 - Ariège
        - 12 - Aveyron  
        - 31 - Haute-Garonne
        - 32 - Gers
        - 46 - Lot
        - 65 - Hautes-Pyrénées
        - 81 - Tarn
        - 82 - Tarn-et-Garonne
        
        *Ancien Languedoc-Roussillon:*
        - 11 - Aude
        - 30 - Gard
        - 34 - Hérault
        - 48 - Lozère
        - 66 - Pyrénées-Orientales
        """)
    
    # Élections disponibles
    st.markdown("---")
    st.subheader("📊 Élections disponibles")
    
    elections = get_available_elections()
    if not elections.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(elections, use_container_width=True)
        
        with col2:
            # Graphique des élections
            fig = px.bar(
                elections, 
                x='annee', 
                y='tour',
                title="Tours d'élection par année",
                labels={'tour': 'Tours disponibles', 'annee': 'Année'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Top nuances politiques
    st.markdown("---")
    st.subheader("🎭 Principales nuances politiques")
    
    nuances_info = df['nuance'].value_counts().head(15)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.bar(
            x=nuances_info.values,
            y=nuances_info.index,
            orientation='h',
            title="Top 15 des nuances (occurrences)",
            labels={'x': 'Nombre d\'occurrences', 'y': 'Nuance politique'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tableau détaillé
        nuances_df = pd.DataFrame({
            'Nuance': nuances_info.index,
            'Occurrences': nuances_info.values,
            'Pourcentage': (nuances_info.values / nuances_info.sum() * 100).round(1)
        })
        st.dataframe(nuances_df, use_container_width=True)

def show_donnees_brutes(df, df_processed, processor):
    """Page d'exploration des données brutes"""
    st.header("📊 Exploration des Données Brutes")
    
    # Filtres
    st.subheader("🔍 Filtres")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        annees_dispo = sorted(df['annee'].unique(), reverse=True)
        annee_selected = st.selectbox("📅 Année", annees_dispo)
    
    with col2:
        tours_dispo = sorted(df[df['annee'] == annee_selected]['tour'].unique())
        tour_selected = st.selectbox("🗳️ Tour", tours_dispo)
    
    with col3:
        depts_dispo = sorted(df['departement'].unique())
        dept_selected = st.selectbox("🏛️ Département", ["Tous"] + depts_dispo)
    
    # Filtrage des données
    data_filtered = df_processed[
        (df_processed['annee'] == annee_selected) & 
        (df_processed['tour'] == tour_selected)
    ]
    
    if dept_selected != "Tous":
        data_filtered = data_filtered[data_filtered['departement'] == dept_selected]
    
    # Métriques de l'élection sélectionnée
    st.markdown("---")
    st.subheader(f"📈 Métriques pour {annee_selected} - Tour {tour_selected}")
    
    if not data_filtered.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Inscrits", f"{data_filtered['inscrits'].sum():,}")
        with col2:
            st.metric("✅ Votants", f"{data_filtered['votants'].sum():,}")
        with col3:
            participation = (data_filtered['votants'].sum() / data_filtered['inscrits'].sum()) * 100
            st.metric("📊 Participation", f"{participation:.1f}%")
        with col4:
            st.metric("🎭 Candidats/Listes", data_filtered['nuance'].nunique())
        
        # Tableau des données détaillées
        st.markdown("---")
        st.subheader("📋 Détail des résultats")
        
        # Préparation du tableau d'affichage
        display_data = data_filtered[
            ['departement', 'nuance', 'voix', 'part_voix', 'taux_participation']
        ].copy()
        
        display_data['part_voix'] = display_data['part_voix'].apply(lambda x: f"{x:.1%}")
        display_data['taux_participation'] = display_data['taux_participation'].apply(lambda x: f"{x:.1%}")
        display_data = display_data.sort_values(['departement', 'voix'], ascending=[True, False])
        
        # Affichage avec possibilité de recherche
        search_term = st.text_input("🔍 Rechercher une nuance:", "")
        if search_term:
            display_data = display_data[display_data['nuance'].str.contains(search_term, case=False, na=False)]
        
        st.dataframe(display_data, use_container_width=True, height=400)
        
        # Graphiques de répartition
        st.markdown("---")
        st.subheader("📊 Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if dept_selected == "Tous":
                # Agrégation par nuance au niveau régional
                agg_data = data_filtered.groupby('nuance')['voix'].sum().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=agg_data.index,
                    y=agg_data.values,
                    title=f"Top 10 des nuances - {annee_selected} Tour {tour_selected}",
                    labels={'x': 'Nuance', 'y': 'Nombre de voix'}
                )
                fig.update_xaxes(tickangle=45)
            else:
                # Résultats pour le département sélectionné
                dept_data = data_filtered[data_filtered['departement'] == dept_selected]
                dept_data = dept_data.sort_values('voix', ascending=False).head(15)
                
                fig = px.bar(
                    dept_data,
                    x='nuance',
                    y='voix',
                    title=f"Résultats département {dept_selected} - {annee_selected} Tour {tour_selected}",
                    labels={'nuance': 'Nuance', 'voix': 'Nombre de voix'}
                )
                fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique en secteurs
            if dept_selected == "Tous":
                pie_data = data_filtered.groupby('nuance')['voix'].sum().sort_values(ascending=False).head(8)
            else:
                pie_data = data_filtered[data_filtered['departement'] == dept_selected]['voix'].head(8)
                pie_data.index = data_filtered[data_filtered['departement'] == dept_selected]['nuance'].head(8)
            
            fig = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title="Répartition des voix (Top 8)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        st.markdown("---")
        st.subheader("📊 Statistiques descriptives")
        
        numeric_cols = data_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = data_filtered[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
    
    else:
        st.warning("⚠️ Aucune donnée disponible pour les filtres sélectionnés.")

def show_analyses_departement(df_processed, processor):
    """Analyses par département"""
    st.header("🏛️ Analyses par Département")
    
    # Sélection de l'élection
    col1, col2 = st.columns(2)
    with col1:
        annees_dispo = sorted(df_processed['annee'].unique(), reverse=True)
        annee_selected = st.selectbox("📅 Année", annees_dispo, key="dept_annee")
    with col2:
        tours_dispo = sorted(df_processed[df_processed['annee'] == annee_selected]['tour'].unique())
        tour_selected = st.selectbox("🗳️ Tour", tours_dispo, key="dept_tour")
    
    # Résumé par département
    dept_summary = processor.create_department_summary(df_processed, annee_selected, tour_selected)
    winners = processor.get_winner_by_department(df_processed, annee_selected, tour_selected)
    
    if not dept_summary.empty and not winners.empty:
        dept_summary = dept_summary.merge(winners, on='departement', how='left')
        
        st.markdown("---")
        st.subheader("📊 Résumé par département")
        
        # Formatage pour affichage
        display_summary = dept_summary.copy()
        display_summary['taux_participation'] = display_summary['taux_participation'].apply(lambda x: f"{x:.1%}")
        display_summary['taux_abstention'] = display_summary['taux_abstention'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_summary, use_container_width=True)
        
        # Visualisations
        st.markdown("---")
        st.subheader("📈 Visualisations par département")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Taux de participation
            fig_participation = px.bar(
                dept_summary,
                x='departement',
                y='taux_participation',
                title="Taux de participation par département",
                labels={'taux_participation': 'Taux de participation', 'departement': 'Département'}
            )
            fig_participation.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig_participation, use_container_width=True)
        
        with col2:
            # Gagnant par département
            fig_winner = px.bar(
                dept_summary,
                x='departement',
                y='voix_gagnantes',
                color='nuance_gagnante',
                title="Candidat/Parti gagnant par département",
                labels={'voix_gagnantes': 'Voix du gagnant', 'departement': 'Département'}
            )
            st.plotly_chart(fig_winner, use_container_width=True)
        
        # Carte de chaleur des résultats
        st.markdown("---")
        st.subheader("🌡️ Répartition des principales nuances par département")
        
        main_nuances = df_processed['nuance'].value_counts().head(8).index.tolist()
        comparison_matrix = processor.create_comparison_matrix(
            df_processed[
                (df_processed['annee'] == annee_selected) & 
                (df_processed['tour'] == tour_selected)
            ],
            main_nuances
        )
        
        if not comparison_matrix.empty:
            fig_heatmap = px.imshow(
                comparison_matrix.T,
                labels=dict(x="Département", y="Nuance", color="Part des voix"),
                title="Répartition des voix par département et nuance",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Analyse comparative
        st.markdown("---")
        st.subheader("🔍 Analyse comparative")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top et bottom participation
            top_participation = dept_summary.nlargest(3, 'taux_participation')[['departement', 'taux_participation']]
            st.write("**🏆 Top 3 participation:**")
            for _, row in top_participation.iterrows():
                st.write(f"- {row['departement']}: {row['taux_participation']:.1%}")
        
        with col2:
            # Répartition des gagnants
            winner_counts = dept_summary['nuance_gagnante'].value_counts()
            st.write("**🎯 Nuances gagnantes:**")
            for nuance, count in winner_counts.items():
                st.write(f"- {nuance}: {count} département(s)")
    
    else:
        st.warning("⚠️ Aucune donnée disponible pour cette élection.")

def show_evolution_temporelle(df_processed, processor):
    """Analyse de l'évolution temporelle"""
    st.header("📈 Évolution Temporelle")
    
    # Sélection de la nuance à analyser
    nuances_dispo = df_processed['nuance'].value_counts().head(20).index.tolist()
    nuance_selected = st.selectbox("🎭 Nuance politique à analyser", nuances_dispo)
    
    # Analyse de l'évolution
    evolution_data = processor.get_evolution_trends(df_processed, nuance_selected)
    
    if not evolution_data.empty:
        st.markdown("---")
        st.subheader(f"📊 Évolution de {nuance_selected}")
        
        # Métriques d'évolution
        if 'evolution_part' in evolution_data.columns:
            avg_evolution = evolution_data['evolution_part'].mean()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Évolution moyenne", f"{avg_evolution:+.1%}")
            with col2:
                best_dept = evolution_data.loc[evolution_data['evolution_part'].idxmax(), 'departement']
                best_evolution = evolution_data['evolution_part'].max()
                st.metric("🏆 Meilleure progression", f"{best_dept}: +{best_evolution:.1%}")
            with col3:
                worst_dept = evolution_data.loc[evolution_data['evolution_part'].idxmin(), 'departement']
                worst_evolution = evolution_data['evolution_part'].min()
                st.metric("📉 Plus forte baisse", f"{worst_dept}: {worst_evolution:.1%}")
        
        # Graphique d'évolution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                evolution_data,
                x='annee',
                y='part_voix',
                color='departement',
                title=f"Évolution de la part de voix de {nuance_selected}",
                labels={'part_voix': 'Part des voix', 'annee': 'Année'}
            )
            fig.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Évolution en nombre de voix
            fig_voix = px.line(
                evolution_data,
                x='annee',
                y='voix',
                color='departement',
                title=f"Évolution du nombre de voix de {nuance_selected}",
                labels={'voix': 'Nombre de voix', 'annee': 'Année'}
            )
            st.plotly_chart(fig_voix, use_container_width=True)
        
        # Tableau d'évolution
        st.markdown("---")
        st.subheader("📋 Détail de l'évolution par département")
        
        evolution_display = evolution_data.pivot(
            index='departement', 
            columns='annee', 
            values='part_voix'
        ).fillna(0)
        
        # Format en pourcentage
        for col in evolution_display.columns:
            evolution_display[col] = evolution_display[col].apply(lambda x: f"{x:.1%}" if x > 0 else "")
        
        st.dataframe(evolution_display, use_container_width=True)
    
    # Comparaison entre nuances
    st.markdown("---")
    st.subheader("🔄 Comparaison entre nuances")
    
    nuances_comparison = st.multiselect(
        "Sélectionner les nuances à comparer",
        nuances_dispo,
        default=nuances_dispo[:3] if len(nuances_dispo) >= 3 else nuances_dispo
    )
    
    if nuances_comparison:
        comparison_data = []
        for nuance in nuances_comparison:
            nuance_data = df_processed[df_processed['nuance'] == nuance].groupby('annee').agg({
                'voix': 'sum',
                'part_voix': 'mean'
            }).reset_index()
            nuance_data['nuance'] = nuance
            comparison_data.append(nuance_data)
        
        if comparison_data:
            comparison_df = pd.concat(comparison_data, ignore_index=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_comparison = px.line(
                    comparison_df,
                    x='annee',
                    y='voix',
                    color='nuance',
                    title="Évolution du nombre de voix par nuance",
                    labels={'voix': 'Nombre de voix', 'annee': 'Année'}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col2:
                fig_part = px.line(
                    comparison_df,
                    x='annee',
                    y='part_voix',
                    color='nuance',
                    title="Évolution de la part de voix par nuance",
                    labels={'part_voix': 'Part des voix', 'annee': 'Année'}
                )
                fig_part.update_yaxes(tickformat='.1%')
                st.plotly_chart(fig_part, use_container_width=True)

def show_predictions(df_processed, processor):
    """Interface de prédictions"""
    st.header("🤖 Prédictions Électorales")
    
    # Chargement du modèle
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Aucun modèle entraîné disponible.")
        st.info("💡 Lancez `python main.py --train` pour entraîner un modèle.")
        
        # Interface de prédiction simulée
        st.markdown("---")
        st.subheader("🎮 Simulateur de prédiction")
        st.info("Interface de démonstration avec prédictions simulées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            taux_participation = st.slider("📊 Taux de participation estimé", 0.4, 0.9, 0.7, 0.01)
            croissance_economique = st.slider("💹 Croissance économique (%)", -3.0, 5.0, 1.0, 0.1)
            taux_chomage = st.slider("💼 Taux de chômage (%)", 5.0, 15.0, 8.0, 0.1)
        
        with col2:
            satisfaction_gouvernement = st.slider("😊 Satisfaction gouvernement", 0.2, 0.8, 0.4, 0.01)
            urbanisation = st.slider("🏙️ Taux d'urbanisation", 0.3, 0.9, 0.6, 0.01)
            age_median = st.slider("👥 Âge médian", 35, 55, 42, 1)
        
        if st.button("🔮 Générer une prédiction simulée", type="primary"):
            # Prédiction simulée
            dept_predictions = {}
            
            for dept in sorted(df_processed['departement'].unique()):
                # Simulation basée sur les paramètres
                base_score = np.random.uniform(0.3, 0.7)
                
                # Ajustements
                if taux_participation > 0.75:
                    base_score += 0.05
                if taux_chomage < 7:
                    base_score += 0.03
                if satisfaction_gouvernement > 0.5:
                    base_score += 0.04
                if croissance_economique > 2:
                    base_score += 0.02
                
                dept_predictions[dept] = min(max(base_score, 0.1), 0.9)
            
            # Affichage des résultats
            st.success("✅ Prédictions générées !")
            
            pred_df = pd.DataFrame(list(dept_predictions.items()), 
                                 columns=['Département', 'Score_Majoritaire_Simulé'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Prédictions par département")
                pred_df['Score_Formaté'] = pred_df['Score_Majoritaire_Simulé'].apply(lambda x: f"{x:.1%}")
                st.dataframe(pred_df[['Département', 'Score_Formaté']], use_container_width=True)
            
            with col2:
                # Graphique des prédictions
                fig = px.bar(
                    pred_df,
                    x='Département',
                    y='Score_Majoritaire_Simulé',
                    title="Prédictions simulées par département",
                    labels={'Score_Majoritaire_Simulé': 'Score prédit'}
                )
                fig.update_yaxes(tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.success("✅ Modèle chargé avec succès !")
        
        # Informations sur le modèle
        model_summary = model.get_model_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🤖 Type de modèle", model_summary.get('model_name', 'N/A'))
        with col2:
            st.metric("🎯 Score CV", f"{model_summary.get('best_score', 0):.1%}")
        with col3:
            st.metric("📊 Variables", model_summary.get('feature_count', 0))
        
        # Interface de prédiction réelle
        st.markdown("---")
        st.subheader("🔮 Interface de prédiction")
        
        try:
            # Préparer des données pour la prédiction
            X, y = processor.prepare_ml_features(df_processed)
            
            if not X.empty:
                st.info("🎮 Ajustez les paramètres ci-dessous pour faire une prédiction")
                
                # Interface utilisateur pour les features
                feature_values = {}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'taux_participation' in X.columns:
                        feature_values['taux_participation'] = st.slider(
                            "📊 Taux de participation", 0.4, 0.9, X['taux_participation'].mean(), 0.01
                        )
                    
                    for col in X.columns:
                        if 'part_' in col and col not in feature_values:
                            nuance = col.replace('part_', '')
                            feature_values[col] = st.slider(
                                f"🎭 Part {nuance}", 0.0, 0.5, X[col].mean(), 0.01
                            )
                
                with col2:
                    for col in X.columns:
                        if col not in feature_values and col in ['taux_abstention', 'inscrits', 'votants']:
                            if 'taux' in col:
                                feature_values[col] = st.slider(
                                    f"📈 {col.replace('_', ' ').title()}", 
                                    0.0, 1.0, X[col].mean(), 0.01
                                )
                            else:
                                feature_values[col] = st.number_input(
                                    f"👥 {col.replace('_', ' ').title()}", 
                                    int(X[col].min()), int(X[col].max()), int(X[col].mean())
                                )
                
                if st.button("🔮 Prédire", type="primary"):
                    try:
                        # Créer le DataFrame d'entrée
                        input_data = pd.DataFrame([feature_values])
                        
                        # Réorganiser selon l'ordre des colonnes du modèle
                        input_data = input_data.reindex(X.columns, axis=1, fill_value=0)
                        
                        # Prédiction
                        predictions, probabilities = model.predict(input_data)
                        
                        # Affichage des résultats
                        st.success("✅ Prédiction réalisée !")
                        
                        if hasattr(model, 'label_encoders') and 'winner' in model.label_encoders:
                            predicted_winner = model.label_encoders['winner'].inverse_transform(predictions)[0]
                            st.metric("🏆 Gagnant prédit", predicted_winner)
                        else:
                            st.metric("🎯 Prédiction", predictions[0])
                        
                        if probabilities is not None:
                            st.subheader("📊 Probabilités détaillées")
                            prob_df = pd.DataFrame({
                                'Classe': range(len(probabilities[0])),
                                'Probabilité': probabilities[0]
                            })
                            prob_df['Probabilité_Format'] = prob_df['Probabilité'].apply(lambda x: f"{x:.1%}")
                            
                            fig_prob = px.bar(
                                prob_df,
                                x='Classe',
                                y='Probabilité',
                                title="Distribution des probabilités"
                            )
                            fig_prob.update_yaxes(tickformat='.1%')
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction: {e}")
            else:
                st.error("❌ Impossible de préparer les données pour la prédiction.")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des features: {e}")

def show_comparaisons(df_processed, processor):
    """Comparaisons entre élections"""
    st.header("🔄 Comparaisons Inter-Élections")
    
    elections_dispo = get_available_elections()
    
    if not elections_dispo.empty:
        elections_list = [f"{row['annee']} - Tour {row['tour']}" 
                        for _, row in elections_dispo.iterrows()]
        
        if len(elections_list) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                election1 = st.selectbox("🗳️ Première élection", elections_list)
                annee1, tour1 = election1.split(" - Tour ")
                annee1, tour1 = int(annee1), int(tour1)
            
            with col2:
                election2 = st.selectbox("🗳️ Deuxième élection", elections_list, 
                                       index=1 if len(elections_list) > 1 else 0)
                annee2, tour2 = election2.split(" - Tour ")
                annee2, tour2 = int(annee2), int(tour2)
            
            if annee1 != annee2 or tour1 != tour2:
                # Données des deux élections
                data1 = processor.create_department_summary(df_processed, annee1, tour1)
                data2 = processor.create_department_summary(df_processed, annee2, tour2)
                
                if not data1.empty and not data2.empty:
                    # Comparaison des participations
                    comparison = data1[['departement', 'taux_participation']].merge(
                        data2[['departement', 'taux_participation']],
                        on='departement',
                        suffixes=(f'_{annee1}', f'_{annee2}')
                    )
                    comparison['evolution_participation'] = (
                        comparison[f'taux_participation_{annee2}'] - 
                        comparison[f'taux_participation_{annee1}']
                    )
                    
                    st.markdown("---")
                    st.subheader("📊 Évolution de la participation")
                    
                    # Métriques globales
                    avg_evolution = comparison['evolution_participation'].mean()
                    max_evolution = comparison['evolution_participation'].max()
                    min_evolution = comparison['evolution_participation'].min()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📈 Évolution moyenne", f"{avg_evolution:+.1%}")
                    with col2:
                        best_dept = comparison.loc[comparison['evolution_participation'].idxmax(), 'departement']
                        st.metric("🏆 Meilleure progression", f"{best_dept}: +{max_evolution:.1%}")
                    with col3:
                        worst_dept = comparison.loc[comparison['evolution_participation'].idxmin(), 'departement']
                        st.metric("📉 Plus forte baisse", f"{worst_dept}: {min_evolution:.1%}")
                    
                    # Graphiques de comparaison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            comparison,
                            x='departement',
                            y='evolution_participation',
                            title=f"Évolution participation: {election1} vs {election2}",
                            labels={'evolution_participation': 'Évolution', 'departement': 'Département'},
                            color='evolution_participation',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_yaxes(tickformat='.1%')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Scatter plot comparaison
                        fig_scatter = px.scatter(
                            comparison,
                            x=f'taux_participation_{annee1}',
                            y=f'taux_participation_{annee2}',
                            text='departement',
                            title=f"Participation {annee1} vs {annee2}",
                            labels={
                                f'taux_participation_{annee1}': f'Participation {annee1}',
                                f'taux_participation_{annee2}': f'Participation {annee2}'
                            }
                        )
                        # Ligne de référence y=x
                        fig_scatter.add_shape(
                            type="line",
                            x0=comparison[f'taux_participation_{annee1}'].min(),
                            y0=comparison[f'taux_participation_{annee1}'].min(),
                            x1=comparison[f'taux_participation_{annee1}'].max(),
                            y1=comparison[f'taux_participation_{annee1}'].max(),
                            line=dict(dash="dash", color="red")
                        )
                        fig_scatter.update_xaxes(tickformat='.1%')
                        fig_scatter.update_yaxes(tickformat='.1%')
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Tableau de comparaison détaillé
                    st.markdown("---")
                    st.subheader("📋 Tableau comparatif détaillé")
                    
                    display_comparison = comparison.copy()
                    for col in [f'taux_participation_{annee1}', f'taux_participation_{annee2}', 'evolution_participation']:
                        display_comparison[col] = display_comparison[col].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(display_comparison, use_container_width=True)
                    
                    # Analyse des gagnants si disponible
                    winners1 = processor.get_winner_by_department(df_processed, annee1, tour1)
                    winners2 = processor.get_winner_by_department(df_processed, annee2, tour2)
                    
                    if not winners1.empty and not winners2.empty:
                        st.markdown("---")
                        st.subheader("🏆 Comparaison des gagnants")
                        
                        winners_comparison = winners1.merge(
                            winners2, 
                            on='departement', 
                            suffixes=(f'_{annee1}', f'_{annee2}')
                        )
                        
                        # Départements qui ont changé de gagnant
                        changed_winners = winners_comparison[
                            winners_comparison[f'nuance_gagnante_{annee1}'] != winners_comparison[f'nuance_gagnante_{annee2}']
                        ]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**🔄 Départements ayant changé ({len(changed_winners)}):**")
                            for _, row in changed_winners.iterrows():
                                st.write(f"- **{row['departement']}**: {row[f'nuance_gagnante_{annee1}']} → {row[f'nuance_gagnante_{annee2}']}")
                        
                        with col2:
                            # Stabilité par nuance
                            stable_depts = len(winners_comparison) - len(changed_winners)
                            stability_rate = stable_depts / len(winners_comparison) if len(winners_comparison) > 0 else 0
                            
                            st.metric("🎯 Taux de stabilité", f"{stability_rate:.1%}")
                            st.write(f"**📊 Départements stables:** {stable_depts}/{len(winners_comparison)}")
                
                else:
                    st.warning("⚠️ Données insuffisantes pour la comparaison.")
            else:
                st.info("ℹ️ Veuillez sélectionner deux élections différentes.")
        else:
            st.warning("⚠️ Pas assez d'élections disponibles pour une comparaison.")
    else:
        st.error("❌ Aucune élection disponible pour la comparaison.")

if __name__ == "__main__":
    main()