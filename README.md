# ⚙️ Steel Weld Quality Prediction using Machine Learning

Project developed as part of the **Machine Learning (3IF3010)** course, aiming to analyze, model, and infer patterns determining steel weld quality using **Supervised and Semi-Supervised Machine Learning** approaches.

---

## 📌 Context and Motivation

The integrity and quality of welded joints are critical factors in heavy industry and renewable energy sectors (such as the manufacturing and welding of pipes for wind turbines and offshore structures), involving multi-billion dollar investments. Traditionally, much of the metallurgical knowledge and quality control relies on the empirical experience of experts or costly destructive laboratory mechanical tests.

This project applies Data Science and Machine Learning techniques to the **Weld Database** to predict mechanical properties and weld quality based on chemical composition parameters and thermal process variables.

---

## 🗂️ Repository Structure

```plaintext
ml_base/
├── data/
│   └── welddb.data         # Raw dataset containing welding data (Weld Database)
├── ML_Project.ipynb        # Jupyter Notebook with the complete ML pipeline and analyses
├── requirements.txt        # Project dependencies and Python libraries
└── README.md               # Project documentation
```

---

## 🔬 Project Pipeline

The study in the [`ML_Project.ipynb`](file:///root/mentionIA/ml_base/ML_Project.ipynb) notebook is structured into the following stages:

### 1. Data Cleaning and Preprocessing
- Mapping and renaming of over 40 metallurgical, operational, and microstructural attributes.
- Handling of special characters and missing values ​​(`'N'` $\to$ `NaN`). - Conversion of data types to numerical formats and discarding of variables with a high rate of missing data (> 50%).

### 2. Exploratory Analysis and Preprocessing
- Descriptive statistical analysis of process parameters.
- **Standardization (Z-Score / `StandardScaler`)**: Necessary due to the wide disparity in physical quantities and orders of magnitude (e.g., chemical compositions in `%` or `ppm`, current in `A`, voltage in `V`, heat input in `kJ/mm`, and temperatures in `°C`).

### 3. Dimensionality Reduction (PCA)
- Application of **Principal Component Analysis (PCA)** to evaluate cumulative variance and the intrinsic dimensionality of the welding data, identifying the number of components required to represent $\ge 90\%$ of the total variance.

### 4. Supervised Learning (Regression)
- **Models**: *Random Forest Regressor* and *XGBoost Regressor*.
- **Target**: Charpy impact transition temperature (`Charpy_temp_C`).
- **Validation**: Rigorous $K$-Fold cross-validation protocol ($K=5$).
- **Metrics**: Coefficient of Determination ($R^2$) and Root Mean Square Error (RMSE).

### 5. Semi-Supervised Learning (Classification)
- **Motivation**: To simulate industrial scenarios where labeled laboratory tests are scarce, while raw process sensor data (unlabeled) is abundant.
- **Approach**: Binarization of the target into quality classes (*High Quality* vs. *Low Quality*) and application of the **Self-Training** algorithm (`SelfTrainingClassifier` using *Random Forest* as the base estimator and iterative pseudo-labeling based on a confidence threshold). - **Metrics**: Accuracy, Precision, Recall, and F1-Score.

### 6. Variable Importance and Metallurgical Conclusions
- Extraction of variable importance rankings (*Feature Importances*).
- Key metallurgical insights:
- **Chemical Composition**: Elements such as Carbon (`Carbon_C`) and Manganese (`Manganese_Mn`) strongly influence the formation of microstructural phases (martensite vs. bainite), thereby determining impact toughness. 
- **Thermal Parameters**: A precise balance between heat input (`Heat_input_kJ_mm`) and current (`Current_A`) is required to avoid excessive widening of the Heat-Affected Zone (HAZ).

---

## 📊 Key Results

| Approach | Model / Algorithm | Primary Metric | Performance |
| :--- | :--- | :--- | :--- |
| **Supervised (Regression)** | Random Forest Regressor | $R^2$ / RMSE | $R^2 \approx 0.78$ \| $\text{RMSE} \approx 13.59$ |
| **Supervised (Regression)** | XGBoost Regressor | $R^2$ / RMSE | $R^2 \approx 0.79$ \| $\text{RMSE} \approx 13.20$ |
| **Semi-Supervised** | Self-Training (Random Forest) | Accuracy / F1-Score | Accuracy $\approx 76\%$ \| F1-Score $\approx 0.76$ | ---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.10 or higher
- `pip` package manager
- Virtual environment (`venv` recommended)

### Step-by-Step

1. **Clone the repository:**
```bash
git clone <REPOSITORY_URL>
cd ml_base
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate # Linux / macOS
# On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook / JupyterLab:**
```bash
jupyter notebook ML_Project.ipynb
```
*Or open the file directly in your preferred editor (such as VS Code or PyCharm with the Jupyter extension).*

---

## 📦 Key Technologies and Libraries

- [Python 3](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — Data manipulation and preprocessing
- [Scikit-Learn](https://scikit-learn.org/) — Standardization, PCA, cross-validation, Machine Learning models, and Self-Training
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient Boosting algorithms
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — Data visualization and feature importance plots

---

## 📄 License

This project was developed for academic and research purposes within the context of the Machine Learning course.