# Audit comparatif du projet de soudures

Revue et améliorations : **El Houssine KAMILI**, `elhoussine-arise`, 5 octobre 2026.

Version examinée : [`8b8c192`](https://github.com/andredemedeiros/weld-quality-prediction/tree/8b8c1928ceae0b0f4494484907e64a6f49d94dc7).
Le nettoyage a été réexécuté. Les 1 652 × 44 champs des données sont identiques
à ceux de l'étude préparée dans cette conversation le 3 octobre.

## Comparaison et priorités

| Point | Dépôt initial | Étude du 3 octobre | Correction dans la branche |
|---|---|---|---|
| Conservation | 307 lignes complètes | 1 652 lignes, labels par cible | Imputer les entrées par pli, conserver les absences de labels |
| Charpy | Température, énergies différentes | Énergie à T connue | Énergie à T connue + température du point à 100 J |
| Cibles | Index positionnels changés entre cellules | Cinq propriétés nommées | Six cibles explicites |
| Chaînes spéciales | Conversion silencieuse en NaN | Traitement documenté | Censure, total d'azote, intervalles, audit |
| Catégories | Supprimées par conversion numérique | One-hot | Procédés harmonisés, AC/DC et polarité conservés |
| Prétraitement | Scaler sur toute la base | Fit par entraînement | Pipeline par pli, ACP descriptive séparée |
| Validation | KFold aléatoire par ligne | CV imbriquée groupée 5 × 3 | Conserver la CV imbriquée par composition |
| Modèles | RF et XGBoost | Huit approches sans XGBoost | Neuf approches, dont XGBoost conservé |
| Semi-supervisé | Zéro U, warning du notebook | RFF graphe et ablation | Graphe + vrai self-training et contrôle supervisé |
| Qualité binaire | Médiane globale, haute température nommée haute qualité | Pas de conformité arbitraire | Classe relative, seuil calculé sur labels visibles d'entraînement |
| Interprétation | Importances internes et recommandations causales | Associations et limites | Permutation externe et conclusions prudentes |
| Reproduction | Liens locaux, chemin ml_base, pip freeze | Scripts, versions, sorties | Liens relatifs et dépendances directes testées |

## Défauts confirmés dans le notebook initial

Les numéros suivants sont les indices JSON, en commençant à zéro.

1. **Cellule 3.** `to_numeric(errors='coerce')` appliqué à toutes les colonnes
   efface les catégories et les notations particulières. Le seuil de présence
   50 % supprime aussi les cibles de traction et ductilité. Puis `dropna()`
   réduit la base à **307 lignes**, sans justification du biais de sélection.
2. **Cellules 3 et 5.** `Quality_class` est d'abord fondée sur `Charpy_temp_C`,
   puis sur `Charpy_toughness_J` via les index des colonnes. La classification
   finale repart sur la température. L'ordre d'exécution modifie donc le sens
   des statistiques intermédiaires.
3. **Cellule 7.** Le scaler est ajusté avant les splits. Ce protocole est
   incorrect pour une comparaison générale; pour les seuls arbres actuels,
   le scaling affine ne modifie pas beaucoup les partitions. Ce défaut ne
   prouve pas à lui seul une inflation de leurs scores.
4. **Cellules 11 et 15.** Le notebook affiche 307 labels et **0 non-label**,
   puis `y contains no unlabeled samples`. L'accuracy de 0,76 sur 62 tests
   est celle d'une classification supervisée; elle ne démontre aucun gain
   semi-supervisé ni utilisation effective de données SCADA.
5. **Cellule 13.** Les essais partageant une composition peuvent se trouver
   dans l'entraînement et le test de KFold. Ce split ne mesure pas une
   généralisation à une nouvelle composition.
6. **Cellule 15.** La médiane est calculée avant de séparer le test. Elle n'est
   pas un seuil industriel de conformité. Une température haute n'est pas
   intrinsèquement une meilleure propriété; à énergie fixée, l'interprétation
   est différente de celle de l'énergie à température fixée.
7. **Cellule 17 et conclusion.** Les importances internes donnent une dépendance
   prédictive, pas le sens d'un effet ni une intervention causale. La largeur
   de zone affectée thermiquement invoquée n'est pas mesurée par ce jeu.

La documentation Cambridge définit la colonne 35 comme **température d'essai
Charpy**. La base initialement conservée contient notamment 176 points à 100 J
et 40 à 28 J. Elle ne fournit donc pas une température de transition unique
sans expliciter un niveau d'énergie et le protocole d'essai. L'énergie peut
être une condition d'une tâche inverse, mais il faut annoncer cette tâche;
elle ne doit pas être présentée comme un réglage de fabrication.

## Nuances sur l'étude précédente

L'étude du 3 octobre était plus complète, sans constituer une validation
industrielle. La méthode RFF graphe n'améliorait pas son ablation supervisée.
Sa signature de composition reste imparfaite avec les données manquantes,
les familles de sources sont heuristiques, la grille est limitée et aucun
test final par nouvelle campagne n'a été effectué. Le Gantt et les données
d'équipe devaient être complétés. Ces limites restent explicites ici.

Il serait incorrect de comparer directement la RMSE initiale XGBoost
**13,20 °C** à la MAE de forêt **18,26 J** du rapport précédent : cibles,
observations, métriques et validation diffèrent. Le nouveau benchmark compare
chaque méthode sur une même cible et les mêmes plis.

## Expériences réellement semi-supervisées

Le graphe exploite les entrées sans labels de la cible, hors groupes de test
et de validation. La cible supplémentaire T100 est définie seulement lorsque
un point d'essai est observé à 100 J; les autres lignes ne fournissent pas
un label T100. Le modèle n'utilise ni température ni autre mesure mécanique
pour prédire T100. Il ne prétend pas reconstruire une courbe de transition.

Le self-training reprend RF comme dans le notebook initial et le compare
à une RF supervisée, avec les **mêmes labels visibles et tests**. Il conserve
25 %, 50 % ou 100 % des groupes labellisés de chaque entraînement, cache
réellement les autres labels et ajoute les U naturels admissibles. Le seuil
binaire est la médiane des seuls labels visibles, avec classe 1 pour une
température inférieure ou égale. C'est une classe relative illustrative,
jamais un verdict de conformité.

Le seuil de pseudo-étiquetage 0,9 et les hyperparamètres sont fixes. Les
probabilités de forêt ne sont pas présentées comme calibrées. Les sorties
comptent les U fournis, pseudo-labels ajoutés et itérations. Un tirage par pli
et fraction est utilisé : sensibilité exploratoire, pas preuve répétée de
supériorité en régime de faibles labels. Résultats : [RESULTS.md](RESULTS.md).

## Attribution Git

Compte connecté vérifié : `elhoussine-arise`, ID 202655074, nom El Houssine
KAMILI. Le dépôt annonce un droit `push` pour ce compte, mais la connexion
API a refusé la création de contenu avec une erreur 403
« Resource not accessible by integration ». Ce droit du compte ne garantit
donc pas les permissions de la connexion. Les nouveaux commits doivent être
vérifiés par leur auteur GitHub après publication; une mention du nom dans le
README ne suffit pas.
Les deux commits initiaux de `root` ne sont pas réattribués.

Pour les futurs commits locaux de ce dépôt :

```bash
git config user.name "El Houssine KAMILI"
git config user.email "202655074+elhoussine-arise@users.noreply.github.com"
```

Cela ne change pas les auteurs des anciens commits et n'affecte pas les
configurations d'autres projets.

## Références

- [Dictionnaire Cambridge](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html).
- scikit-learn [validation groupée](https://scikit-learn.org/stable/modules/cross_validation.html),
  [prétraitement](https://scikit-learn.org/stable/common_pitfalls.html),
  [SelfTrainingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html).
- [Belkin et al., régularisation de variété](https://www.jmlr.org/papers/v7/belkin06a.html).
- [GitHub, attribution et adresse des commits](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).
