# ⚙️ Steel Weld Quality Prediction using Machine Learning

Project developed for the **Apprentissage Automatique (3IF3010)** course by André Filipe DE MEDEIROS, El Houssine KAMILI and Nicolò DAL MONTE. It analyses the **Weld Database** and predicts weld quality with **supervised and semi-supervised machine learning**.

---

## 📌 Context and Motivation

The quality of welded joints is critical in heavy industry and renewable energy, for example in the welding of tubes for wind turbines. Knowledge of weld quality still relies largely on the experience of specialists and on costly destructive mechanical tests.

This project uses the chemical composition and the welding process parameters of the weld metal to predict its mechanical properties. It then compares the models and turns the results into recommendations for good weld quality.

---

## 🗂️ Repository Structure

```plaintext
ml_base/
├── data/
│   └── welddb.data         # Raw Weld Database (MAP_DATA_WELD, 1652 rows x 44 columns)
├── results/                # Cached outputs of Phase 5 (created when the notebook runs)
├── ML_Project.ipynb        # Complete analysis: EDA, preprocessing, PCA, models, comparison
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔬 Project Pipeline

The notebook [`ML_Project.ipynb`](ML_Project.ipynb) has six phases.

### Phase 1: Exploratory Data Analysis
- Structure, missing values (encoded as `N`) and non-numeric entries (detection limits such as `<5`, ranges such as `150-200`, labels such as `66totndres`).
- Distributions, categorical variables, correlations and candidate quality variables.
- Detection of suspicious repeated Charpy values (356 values of exactly 100 J, 118 of 28 J).
- Identification of weld groups: several rows are tests of the same weld.

### Phase 2: Preprocessing
- **Parsing of non-standard values.** A detection limit `<x` is replaced by half the limit, with unit conversion when it is given in the wrong unit (wt% or ppm). A range is replaced by its midpoint.
- **Feature selection.** We keep the 30 composition and process columns, which are known before the weld is tested. The 5 columns with more than 80% missing values are dropped, leaving **25 features**. Mechanical properties and microstructure are excluded, because they are results of the weld, not inputs.
- **Weld groups** (same composition and process) are defined for group cross-validation. Rare weld types are grouped into `Other`.
- **`ColumnTransformer`, fitted on the training folds only:**
  - alloying elements: imputed to 0;
  - other numerical variables: median imputation;
  - missing indicators added;
  - categorical variables: one-hot encoded;
  - **z-score standardisation**.

### Phase 3: Principal Component Analysis
- The PCA is exploratory only. PC1 is about 20% of the variance (alloy content and heat treatment) and PC2 about 14% (welding energy).
- 14 of 22 components are needed for 90% of the variance, so the data are not low-dimensional.
- The components are **not** used as model inputs. The original variables are kept so the results stay interpretable.

### Phase 4: Quality Variables and Prediction Strategy
Quality has two dimensions that move in opposite directions, so there are two separate models:

| Dimension | Targets | Data |
|---|---|---|
| **Strength** (main) | `Yield_strength_MPa`, `Ultimate_tensile_strength_MPa`, `Elongation_pct` (multi-output) | 665 welds, 462 independent groups |
| **Toughness** (secondary) | `Charpy_toughness_J`, with the test temperature `Charpy_temp_C` as an input | 879 tests, 328 groups |

### Phase 5: Modelling and Validation
- **Nested cross-validation with `GroupKFold`:** 5 outer folds produce out-of-fold predictions, and 3 inner folds tune the hyperparameters. Rows of the same weld (or the same `Weld_ID`) never appear in both training and test.
- **Supervised models:** mean baseline, linear regression, Ridge, KNN, SVR (RBF kernel), decision tree, bagging, random forest and ExtraTrees.
- **Semi-supervised method (strength only):** a graph-regularised regression (manifold regularisation with a Nyström kernel map) uses the 987 welds without complete strength measurements. It is compared with supervised controls on the same folds.
- **Sensitivity analysis:** the toughness task is rerun without the 100 J values.

### Phase 6: Comparative Analysis, Conclusion and Recommendations
- **Metrics:**
  - RMSE in physical units is the main metric, together with MAE, R², normalised RMSE and bias.
  - All scores are weighted by weld group, so they estimate the error on a new weld.
  - Additional baselines: mean per weld type and mean per test temperature.
- **Uncertainty:** differences between models are checked with a paired bootstrap on the weld groups.
- **Error analysis:** errors are broken down by weld type.
- **Interpretation:** permutation importance and partial dependence of the selected models, followed by recommendations.

---

## 📊 Key Results

Out-of-fold scores, weighted by weld group:

| Task | Best model | R² | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- |
| Yield strength | SVR (RBF) | 0.78 | 39 MPa | 25 MPa |
| Tensile strength | SVR (RBF) | 0.88 | 31 MPa | 19 MPa |
| Elongation | SVR (RBF) | 0.74 | 2.6 % | 1.9 % |
| Charpy energy | ExtraTrees | 0.73 | 21 J | 12 J |
| Charpy energy without the 100 J values | ExtraTrees | 0.80 | 22 J | 10 J |

**Model comparison:**
- **Strength:** SVR is first. ExtraTrees comes second; the gap is small but statistically significant.
- **Toughness:** ExtraTrees and random forest are practically equivalent.
- **Linear models and the single tree:** clearly worse. On the Charpy task the linear models extrapolate badly on a family of welds with extreme sulphur and phosphorus contents.
- **Semi-supervised graph regression:** no measurable gain over its supervised control. The unlabeled welds neither help nor hurt.

**Limits:** errors are 2 to 3 times larger for rare weld types than for MMA welds, which make up about 72% of the strength data. Predictions are only reliable within the range of the database.

**Main factors and recommendations:**
- **Strength:** increases with the alloying elements (Cr, Mn, Nb, V, Mo, Ni, C), at the expense of elongation (strength/ductility trade-off).
- **Toughness:** dominated by the test temperature. It decreases with Mo, Cr, P and O.
- **Recommendations:**
  - Treat quality as a strength/toughness compromise.
  - Obtain strength with moderate, combined alloying.
  - Keep impurities (P, S) and oxygen low.
  - Specify the Charpy test at the service temperature.
  - Use the models for screening, with a margin of one RMSE and a warning outside the domain of the data, not as a substitute for testing.

These are associations learned from historical data, not causal effects. They should be confirmed by designed welding trials.

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.10 or higher
- `pip`, and a virtual environment (`venv` recommended)

### Step-by-Step

1. **Clone the repository:**
```bash
git clone <REPOSITORY_URL>
cd ml_base
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux / macOS
# On Windows: .venv\Scripts\activate
```

3. **Install the dependencies:**
```bash
pip install -r requirements.txt
```

4. **Open the notebook** in VS Code or PyCharm (Jupyter extension, using the `.venv` kernel), or with Jupyter:
```bash
pip install notebook
jupyter notebook ML_Project.ipynb
```

The full Phase 5 computation takes about 4 minutes. Its outputs are saved in `results/` and reloaded on later runs, as long as the protocol has not changed. Set `RECOMPUTE = True` in section 5.7 to force a new run.

---

## 📦 Key Technologies and Libraries

- [Python 3](https://www.python.org/)
- [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/): data manipulation
- [scikit-learn](https://scikit-learn.org/): preprocessing pipelines, PCA, group cross-validation, models, permutation importance
- [SciPy](https://scipy.org/): sparse graph and linear algebra of the semi-supervised method
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/): visualisation

---

## 📚 Data Source

Cool, T. and Bhadeshia, H. K. D. H. *MAP Data Library — MAP_DATA_WELD*. [Documentation](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html). The full bibliography (semi-supervised learning, interpretation tools, bootstrap) is at the end of the notebook.

---

## 📄 License

This project was developed for academic and research purposes within the context of the Machine Learning course.
