# MSPR-BigData-MachineLearning

## Description

Ce projet propose une plateforme complète pour l'analyse, la visualisation et la prédiction des résultats électoraux en région Occitanie, dans le cadre de la MSPR Big Data & Machine Learning (TPRE813). Il couvre la collecte de données réelles, leur traitement, la création de modèles prédictifs avancés et la mise à disposition d'une interface web interactive.

## Équipe

- Paul CARION
- Yassin FARASSI
- Mathieu GAISNON
- Julie MONTOUX

## Fonctionnalités

- 🗳️ Collecte et traitement des résultats électoraux pour les 13 départements d'Occitanie via une base MySQL
- 📊 Analyses statistiques, géographiques et corrélationnelles interactives
- 🤖 Prédictions avec modèles de machine learning, basées sur les vraies nuances politiques issues de la BDD
- 📈 Dashboard web Streamlit convivial : exploration, visualisation, ML, prédiction
- 📋 Génération de rapports et comparaisons inter-élections

## Architecture du projet

```
MSPR-BigData-MachineLearning/
├── config/                  # Paramétrage et accès BDD
├── src/
│   ├── data_collection/     # Scripts d'import et requêtes MySQL
│   ├── data_processing/     # Préparation, nettoyage et features
│   ├── models/              # Modèles ML et gestion du pipeline
│   ├── prediction/          # Prédicteur basé sur la BDD, nuances réelles
│   └── visualization/       # Graphiques et analyses avancées
├── data/
│   ├── raw/                 # Données brutes exportées
│   └── processed/           # Données nettoyées/prêtes
├── models/                  # Modèles sauvegardés (joblib)
├── visualizations/          # Graphiques générés
├── main.py                  # Pipeline principal (CLI)
├── app.py                   # Interface Streamlit (web)
├── requirements.txt         # Dépendances Python
└── tests/                   # Tests unitaires et d'intégration
```

## Installation

### 1. Clonage du dépôt

```bash
git clone https://github.com/mgaisnon/MSPR-BigData-MachineLearning.git
cd MSPR-BigData-MachineLearning
```

### 2. Environnement virtuel

```bash
python -m venv venv
source venv/bin/activate         # Linux/Mac
venv\Scripts\activate            # Windows
```

### 3. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration de la base de données

Créez un fichier `.env` à la racine, sur le modèle de `.env.example`, avec vos paramètres MySQL :

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=mot_de_passe
MYSQL_DATABASE=bddelections
```

Assurez-vous que la base et la table `resultatslegi` existent (voir plus bas).

## Utilisation

### Pipeline complet

```bash
python main.py --all
```

### Étapes individuelles

```bash
python main.py --collect    # Collecte des données MySQL
python main.py --train      # Entraînement des modèles ML
python main.py --analyze    # Analyses et visualisations
python main.py --predict    # Génération de prédictions
```

### Interface web

```bash
streamlit run app.py
```

## Structure de la base de données

Table `resultatslegi` :

- `id` : Identifiant unique
- `annee` : Année de l'élection
- `tour` : Tour (1 ou 2)
- `departement` : Code INSEE du département
- `inscrits` : Nombre d'inscrits
- `votants` : Nombre de votants
- `abstentions` : Nombre d'abstentions
- `exprimes` : Nombre de suffrages exprimés
- `nuance` : Nuance politique (parti/coalition réelle)
- `voix` : Nombre de voix

## Départements couverts

**Ancien Midi-Pyrénées** : 09, 12, 31, 32, 46, 65, 81, 82  
**Ancien Languedoc-Roussillon** : 11, 30, 34, 48, 66

## Modèles de Machine Learning

- Régression logistique
- Random Forest
- Gradient Boosting
- XGBoost, LightGBM
- SVM, Naive Bayes, K-NN

Comparaison automatique des modèles sur données réelles (nuances issues de la BDD).

## Analyses disponibles

- Statistiques descriptives, temporelles, par nuance/département
- Corrélations, importance des variables
- Cartes interactives
- Visualisation des performances modèles (accuracy, ROC, importance)

## Interface Streamlit

- 🏠 Accueil
- 📊 Données (exploration, requêtes)
- 📈 Analyses (stats, graphiques, cartes)
- 🤖 Prédictions (simulateur interactif, nuances authentiques)
- 📋 Comparaisons historiques

## Logging

Logs dans `election_predictor.log`  
Niveaux : INFO, WARNING, ERROR

## Tests

```bash
pytest tests/
```

## Contribution

1. Fork
2. Branche feature (`git checkout -b feature/maFeature`)
3. Commit (`git commit -m 'Ajout de ma fonctionnalité'`)
4. Push (`git push origin feature/maFeature`)
5. Pull Request

## Licence

Projet académique - MSPR TPRE813

## Support

Pour toute question, ouvrir une issue ou contacter l'équipe.

## Roadmap

- [ ] Intégration de données socio-économiques
- [ ] Modèles Deep Learning
- [ ] API REST pour prédictions
- [ ] Déploiement cloud
- [ ] Interface mobile