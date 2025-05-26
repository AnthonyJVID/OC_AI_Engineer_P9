# 🚗 Preuve de Concept – SegFormer pour la Segmentation d’Images Urbaines

Ce projet est réalisé dans le cadre du parcours AI Engineer d’OpenClassrooms

Le dépôt présente une **preuve de concept (PoC)** dans le cadre d’un test technique pour l’entreprise **DataSpace**, visant à démontrer la supériorité d’un modèle de segmentation d’images récent, **SegFormer B5**, face à une baseline classique **U-Net (EfficientNetB3)**.

---

## 🎯 Objectif

- **Problématique métier :** Améliorer les performances de la segmentation d’images embarquées dans des véhicules autonomes.
- **Défi technique :** Mettre en œuvre une méthode récente (moins de 5 ans) issue de la veille technologique, pour améliorer vitesse, précision et efficacité mémoire.
- **Modèle évalué :** SegFormer B5 (architecture Transformer) vs U-Net + EfficientNetB3.

---

## 🗂 Structure du dépôt

```
├── notebook.ipynb              ← Comparaison des modèles (baseline vs SegFormer)
├── fonctions.py                ← Fonctions utilitaires : preprocessing, métriques, affichage
├── code_dashboard.py           ← Application Streamlit interactive de comparaison
├── note_methodo.pdf            ← Note méthodologique avec résultats et analyse
├── plan_travail.pdf            ← Plan prévisionnel validé pour la PoC
```

---

## 📚 Données

- **Dataset utilisé :** [Cityscapes](https://www.cityscapes-dataset.com/)
- **Images utilisées :** 2000 (train), 500 (validation)
- **Dimensions :** 256x256 px
- **Classes :** 8 grandes catégories remappées à partir des 34 originales

---

## 🧠 Modèles comparés

| Modèle           | Framework    | Type             | Déploiement Cloud            |
|------------------|--------------|------------------|------------------------------|
| EfficientNetB3   | Keras        | CNN              | **AWS EC2 (FastAPI)**        |
| SegFormer B5     | PyTorch      | Transformer      | **Hugging Face Space**       |

---

## 🧪 Évaluation

- **Métriques** : `Accuracy`, `Dice`, `mIoU`, `Loss`, `Temps d'entraînement`
- **Outils** :
  - Entraînement avec **Albumentations**, `torch.cuda.amp` (mixed precision), `ReduceLROnPlateau`
  - Visualisation des performances par epoch
  - Comparaison via **Streamlit dashboard**
  - Appels API vers modèles déployés sur le cloud

---

## 📊 Résultats

| Modèle        | Accuracy | Dice | mIoU | Loss  | Temps entraînement |
|---------------|----------|------|------|-------|---------------------|
| SegFormer B5  | ~0.90    | 0.83 | 0.70 | ~0.20 | **160.93 min**      |
| U-Net (EffB3) | ~0.91    | 0.86 | 0.67 | ~0.45 | 368.37 min          |

- **SegFormer** offre une **meilleure généralisation** (mIoU) et un **temps d'entraînement bien plus court**
- **U-Net** reste légèrement supérieur en Dice, mais plus coûteux en calcul

---

## 🖥️ Dashboard interactif

Le fichier `code_dashboard.py` fournit une interface **Streamlit** permettant de :

- Comparer les masques réels et prédits pour chaque modèle
- Obtenir les métriques associées (Dice, IoU)
- Consommer dynamiquement les **API de prédiction** :
  - **SegFormer déployé sur Hugging Face**
  - **EfficientNetB3 déployé sur AWS EC2 (FastAPI)**

---

## 📄 Documentation

- `note_methodo.pdf` : Analyse détaillée des résultats, fonctionnement du modèle SegFormer, compréhension des concepts
- `plan_travail.pdf` : Justification du modèle choisi, références bibliographiques, méthode de test

---

## 🔍 Références

- [SegFormer – Article Arxiv](https://arxiv.org/abs/2105.15203)
- [Blog Medium – SegFormer](https://medium.com/geekculture/semantic-segmentation-with-segformer-2501543d2be4)
- [Tutoriel Debugger Café](https://debuggercafe.com/segformer-for-semantic-segmentation)

---

## 👨‍💻 Auteur

Projet réalisé par **AnthonyJVID** dans le cadre de la mission 9 – Preuve de concept IA pour DataSpace.
