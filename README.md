# Modèle Prédictif des Élections - Région Occitanie

## Description

Ce projet développe un système complet d'analyse et de prédiction des résultats électoraux pour la région Occitanie, dans le cadre du cours TPRE813 - Big Data & Business Intelligence.

## Équipe

- Paul CARION
- Yassin FARASSI  
- Mathieu GAISNON
- Julie MONTOUX

## Fonctionnalités

- 🗳️ Collecte et traitement des données électorales des 13 départements d'Occitanie
- 📊 Analyses statistiques et visualisations interactives
- 🤖 Modèles de machine learning pour la prédiction
- 📈 Dashboard web interactif avec Streamlit
- 📋 Génération de rapports automatiques

## Architecture

```
election_predictor_occitanie/
├── config/                 # Configuration
├── src/
│   ├── data_collection/    # Collecte de données
│   ├── data_processing/    # Traitement des données
│   ├── models/            # Modèles ML
│   └── visualization/     # Analyses et visualisations
├── data/
│   ├── raw/               # Données brutes
│   └── processed/         # Données traitées
├── models/                # Modèles entraînés
├── visualizations/        # Graphiques générés
├── main.py               # Pipeline principal
├── app.py                # Interface Streamlit
└── requirements.txt      # Dépendances
```

## Installation

### 1. Cloner le projet
```bash
git clone <repository-url>
cd election_predictor_occitanie
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration de la base de données
```bash
cp .env.example .env
# Éditer .env avec vos paramètres de base de données
```

## Utilisation

### Pipeline complet
```bash
python main.py --all
```

### Étapes individuelles
```bash
# Collecte des données
python main.py --collect

# Entraînement des modèles
python main.py --train

# Analyses et visualisations
python main.py --analyze

# Prédictions
python main.py --predict
```

### Interface web
```bash
streamlit run app.py
```

## Structure de la base de données

Table `resultatslelegi` :
- `id` : Identifiant unique
- `annee` : Année de l'élection
- `tour` : Tour de l'élection (1 ou 2)
- `departement` : Code département
- `inscrits` : Nombre d'inscrits
- `votants` : Nombre de votants
- `abstentions` : Nombre d'abstentions
- `exprimes` : Nombre de suffrages exprimés
- `nuance` : Nuance politique
- `voix` : Nombre de voix

## Départements couverts

**Ancien Midi-Pyrénées :**
- 09 - Ariège
- 12 - Aveyron
- 31 - Haute-Garonne
- 32 - Gers
- 46 - Lot
- 65 - Hautes-Pyrénées
- 81 - Tarn
- 82 - Tarn-et-Garonne

**Ancien Languedoc-Roussillon :**
- 11 - Aude
- 30 - Gard
- 34 - Hérault
- 48 - Lozère
- 66 - Pyrénées-Orientales

## Modèles de Machine Learning

Le système teste plusieurs algorithmes :
- Régression logistique
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- SVM
- Naive Bayes
- K-NN

## Analyses disponibles

1. **Analyses descriptives**
   - Statistiques par département
   - Évolution temporelle
   - Répartition des nuances politiques

2. **Analyses de corrélation**
   - Matrice de corrélation interactive
   - Variables les plus influentes

3. **Analyses géographiques**
   - Cartes des résultats
   - Comparaisons inter-départementales

4. **Analyses de performance**
   - Métriques des modèles
   - Courbes ROC
   - Importance des variables

## Interface Streamlit

L'application web propose :
- 🏠 **Accueil** : Vue d'ensemble du projet
- 📊 **Données** : Exploration des données brutes
- 📈 **Analyses** : Analyses par département et évolution temporelle
- 🤖 **Prédictions** : Interface de prédiction interactive
- 📋 **Comparaisons** : Comparaisons inter-élections

## Logging

Les logs sont sauvegardés dans `election_predictor.log` avec différents niveaux :
- INFO : Informations générales
- WARNING : Avertissements
- ERROR : Erreurs

## Tests

```bash
pytest tests/
```

## Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## Licence

Projet académique - MSPR TPRE813

## Support

Pour toute question, contacter l'équipe de développement.

## Roadmap

- [ ] Intégration de données socio-économiques externes
- [ ] Modèles de deep learning
- [ ] API REST pour les prédictions
- [ ] Déploiement sur cloud
- [ ] Interface mobile