# Résultats exécutés du protocole revu

El Houssine KAMILI, 5 octobre 2026. Graine 42; CV imbriquée 5 plis externes × 3 plis internes par composition.

## Meilleure MAE par cible

| target | model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|---|
| elongation | SVR RBF | 1.883 | 0.141 | 2.691 | 0.695 |
| reduction_area | SVR RBF | 2.489 | 0.527 | 4.596 | 0.717 |
| temperature_100J | SVR RBF | 12.289 | 1.841 | 18.433 | 0.484 |
| charpy_energy | Forêt | 18.260 | 4.320 | 29.071 | 0.648 |
| tensile_strength | SVR RBF | 20.083 | 1.668 | 31.518 | 0.871 |
| yield_strength | SVR RBF | 26.502 | 2.506 | 42.511 | 0.787 |

MAE/RMSE : MPa pour les résistances, points de pourcentage pour la ductilité, J pour l’énergie, °C pour T100. Écarts types de plis, pas intervalles de confiance. R² regroupé sur les prédictions hors pli.

## yield_strength

| model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|
| SVR RBF | 26.502 | 2.506 | 42.511 | 0.787 |
| XGBoost | 28.342 | 3.358 | 42.177 | 0.787 |
| Forêt | 31.459 | 3.075 | 45.353 | 0.754 |
| kNN | 36.317 | 4.196 | 53.218 | 0.666 |
| RFF supervisé | 43.061 | 2.598 | 58.856 | 0.597 |
| RFF graphe | 43.113 | 2.657 | 58.892 | 0.597 |
| Ridge | 45.142 | 3.297 | 64.026 | 0.520 |
| Arbre | 45.500 | 2.566 | 61.899 | 0.551 |
| Moyenne | 71.723 | 6.933 | 92.588 | -0.002 |

## tensile_strength

| model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|
| SVR RBF | 20.083 | 1.668 | 31.518 | 0.871 |
| XGBoost | 23.844 | 1.931 | 35.179 | 0.838 |
| Forêt | 27.505 | 3.216 | 39.310 | 0.797 |
| kNN | 30.321 | 2.876 | 45.490 | 0.728 |
| Ridge | 33.500 | 1.882 | 47.442 | 0.708 |
| RFF supervisé | 35.382 | 3.552 | 51.592 | 0.648 |
| RFF graphe | 35.485 | 3.537 | 51.672 | 0.648 |
| Arbre | 40.153 | 2.109 | 56.618 | 0.590 |
| Moyenne | 70.158 | 6.326 | 88.275 | -0.003 |

## elongation

| model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|
| SVR RBF | 1.883 | 0.141 | 2.691 | 0.695 |
| Forêt | 1.946 | 0.161 | 2.665 | 0.702 |
| XGBoost | 1.969 | 0.168 | 2.697 | 0.694 |
| kNN | 2.177 | 0.216 | 2.945 | 0.632 |
| Ridge | 2.347 | 0.243 | 3.184 | 0.571 |
| RFF supervisé | 2.585 | 0.213 | 3.328 | 0.535 |
| RFF graphe | 2.586 | 0.214 | 3.328 | 0.535 |
| Arbre | 2.734 | 0.399 | 3.625 | 0.440 |
| Moyenne | 4.047 | 0.291 | 4.896 | -0.003 |

## reduction_area

| model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|
| SVR RBF | 2.489 | 0.527 | 4.596 | 0.717 |
| Forêt | 2.640 | 0.592 | 4.690 | 0.698 |
| XGBoost | 2.757 | 0.457 | 4.826 | 0.689 |
| kNN | 2.770 | 0.522 | 4.882 | 0.680 |
| Ridge | 3.157 | 0.543 | 4.872 | 0.684 |
| RFF supervisé | 3.518 | 0.695 | 5.509 | 0.594 |
| RFF graphe | 3.530 | 0.681 | 5.581 | 0.587 |
| Arbre | 3.583 | 0.682 | 6.204 | 0.490 |
| Moyenne | 6.745 | 0.921 | 8.829 | -0.012 |

## charpy_energy

| model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|
| Forêt | 18.260 | 4.320 | 29.071 | 0.648 |
| XGBoost | 19.093 | 2.852 | 29.171 | 0.655 |
| kNN | 22.341 | 4.207 | 34.530 | 0.509 |
| Arbre | 22.799 | 2.448 | 38.973 | 0.388 |
| SVR RBF | 22.974 | 2.871 | 33.238 | 0.549 |
| RFF supervisé | 26.425 | 2.551 | 33.662 | 0.546 |
| RFF graphe | 26.492 | 2.574 | 33.739 | 0.544 |
| Ridge | 32.367 | 8.036 | 49.529 | -0.222 |
| Moyenne | 38.926 | 2.171 | 50.180 | -0.006 |

## temperature_100J

| model | MAE_mean | MAE_sd | RMSE_mean | R2_pooled |
|---|---|---|---|---|
| SVR RBF | 12.289 | 1.841 | 18.433 | 0.484 |
| Forêt | 12.549 | 2.435 | 18.470 | 0.477 |
| XGBoost | 12.979 | 1.646 | 18.813 | 0.462 |
| kNN | 13.429 | 1.896 | 19.319 | 0.437 |
| RFF graphe | 14.691 | 2.517 | 19.667 | 0.411 |
| RFF supervisé | 14.710 | 2.500 | 19.713 | 0.408 |
| Arbre | 15.448 | 2.533 | 21.451 | 0.295 |
| Ridge | 15.902 | 3.058 | 20.931 | 0.329 |
| Moyenne | 19.709 | 2.501 | 25.782 | -0.003 |

## Self-training comparé au supervisé

| fraction_labelled_groups | model | accuracy | balanced_accuracy | F1 | n_pseudo_labels_added |
|---|---|---|---|---|---|
| 0.250 | RF self-training | 0.666 | 0.668 | 0.671 | 552.200 |
| 0.250 | RF supervised | 0.677 | 0.678 | 0.690 | 0.000 |
| 0.500 | RF self-training | 0.719 | 0.726 | 0.732 | 401.000 |
| 0.500 | RF supervised | 0.713 | 0.717 | 0.724 | 0.000 |
| 1.000 | RF self-training | 0.753 | 0.753 | 0.747 | 555.400 |
| 1.000 | RF supervised | 0.772 | 0.773 | 0.770 | 0.000 |

La classe est relative à la médiane des labels visibles d’entraînement de T100. Les probabilités RF ne sont pas calibrées. À 50 % de groupes labellisés conservés, le gain de F1 est modeste (0,732 contre 0,724); à 25 % et 100 %, le self-training est moins bon. Ces résultats ne démontrent pas une supériorité générale; un tirage par pli/fraction seulement.

## Limites

Les groupes sont des signatures observées, pas des lots certifiés indépendants. Les scores sont comparables uniquement sur une même cible et un même protocole. Le modèle de T100 n’est pas une courbe de transition complète. Aucune recette universelle ni conformité industrielle n’est démontrée. Voir l’audit et les CSV pour les sensibilités et paramètres.
