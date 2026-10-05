# ⚙️ Previsão de Qualidade de Soldas em Aços com Machine Learning

Projeto desenvolvido no âmbito da disciplina **Apprentissage Automatique (3IF3010)**, com o objetivo de analisar, modelar e inferir padrões determinantes na qualidade de soldas em aços utilizando abordagens de **Aprendizado de Máquina Supervisionado e Semi-Supervisionado**.

---

## 📌 Contexto e Motivação

A integridade e a qualidade de juntas soldadas são fatores críticos em indústrias pesadas e de energia renovável (como a fabricação e soldagem de tubos para turbinas eólicas e estruturas offshore), envolvendo investimentos bilionários. Tradicionalmente, grande parte do conhecimento metalúrgico e do controle de qualidade depende da experiência empírica de especialistas ou de ensaios mecânicos destrutivos laboratoriais dispendiosos.

Este projeto aplica técnicas de Ciência de Dados e Machine Learning sobre a **Weld Database** para prever propriedades mecânicas e a qualidade da solda a partir de parâmetros de composição química e variáveis térmicas de processo.

---

## 🗂️ Estrutura do Repositório

```plaintext
ml_base/
├── data/
│   └── welddb.data         # Dataset bruto com dados de soldagem (Weld Database)
├── ML_Project.ipynb        # Jupyter Notebook com pipeline completo de ML e análises
├── requirements.txt        # Dependências e bibliotecas Python do projeto
└── README.md               # Documentação do projeto
```

---

## 🔬 Pipeline do Projeto

O estudo no notebook [`ML_Project.ipynb`](file:///root/mentionIA/ml_base/ML_Project.ipynb) está estruturado nas seguintes etapas:

### 1. Limpeza e Tratamento de Dados
- Mapeamento e renomeação de mais de 40 atributos metalúrgicos, operacionais e microestruturais.
- Tratamento de caracteres especiais e valores ausentes (`'N'` $\to$ `NaN`).
- Conversão de tipos de dados para formatos numéricos e descarte de variáveis com alta taxa de dados faltantes (> 50%).

### 2. Análise Exploratória e Pré-processamento
- Análise estatística descritiva dos parâmetros do processo.
- **Padronização (Z-Score / `StandardScaler`)**: Necessária devido à grande disparidade de grandezas e ordens de magnitude físicas (ex.: composições químicas em `%` ou `ppm`, corrente em `A`, tensão em `V`, calor de entrada em `kJ/mm` e temperaturas em `°C`).

### 3. Redução de Dimensionalidade (PCA)
- Aplicação de **Análise de Componentes Principais (PCA)** para avaliar a variância cumulativa e a dimensionalidade intrínseca dos dados de soldagem, identificando o número de componentes necessários para representar $\ge 90\%$ da variância total.

### 4. Aprendizado Supervisionado (Regressão)
- **Modelos**: *Random Forest Regressor* e *XGBoost Regressor*.
- **Alvo**: Temperatura de transição de impacto Charpy (`Charpy_temp_C`).
- **Validação**: Protocolo rigoroso de validação cruzada $K$-Fold ($K=5$).
- **Métricas**: Coeficiente de Determinação ($R^2$) e Raiz do Erro Quadrático Médio (RMSE).

### 5. Aprendizado Semi-Supervisionado (Classificação)
- **Motivação**: Simular cenários industriais onde testes laboratoriais rotulados são escassos, enquanto dados brutos de sensores de processo (não rotulados) são abundantes.
- **Abordagem**: Binarização do alvo em classes de qualidade (*Alta Qualidade* vs *Baixa Qualidade*) e aplicação do algoritmo **Self-Training** (`SelfTrainingClassifier` com *Random Forest* como estimador base e pseudo-rotulagem iterativa baseada em limiar de confiança).
- **Métricas**: Acurácia, Precisão, Recall e F1-Score.

### 6. Importância de Variáveis e Conclusões Metalúrgicas
- Extração do ranking de importância das variáveis (*Feature Importances*).
- Insights metalúrgicos principais:
  - **Composição Química**: Elementos como Carbono (`Carbon_C`) e Manganês (`Manganese_Mn`) influenciam fortemente a formação de fases microestruturais (martensita vs. bainita), determinando a tenacidade ao impacto.
  - **Parâmetros Térmicos**: Equilíbrio estrito entre aporte térmico (`Heat_input_kJ_mm`) e corrente (`Current_A`), evitando o alargamento excessivo da Zona Afetada pelo Calor (ZAC/HAZ).

---

## 📊 Principais Resultados

| Abordagem | Modelo / Algoritmo | Métrica Principal | Desempenho |
| :--- | :--- | :--- | :--- |
| **Supervisionado (Regressão)** | Random Forest Regressor | $R^2$ / RMSE | $R^2 \approx 0.78$ \| $\text{RMSE} \approx 13.59$ |
| **Supervisionado (Regressão)** | XGBoost Regressor | $R^2$ / RMSE | $R^2 \approx 0.79$ \| $\text{RMSE} \approx 13.20$ |
| **Semi-Supervisionado** | Self-Training (Random Forest) | Acurácia / F1-Score | Acurácia $\approx 76\%$ \| F1-Score $\approx 0.76$ |

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10 ou superior
- Gerenciador de pacotes `pip`
- Ambiente virtual (`venv` recomendado)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd ml_base
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   # No Windows: .venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Inicie o Jupyter Notebook / JupyterLab:**
   ```bash
   jupyter notebook ML_Project.ipynb
   ```
   *Ou abra o arquivo diretamente em seu editor de preferência (como VS Code ou PyCharm com extensão Jupyter).*

---

## 📦 Principais Tecnologias e Bibliotecas

- [Python 3](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — Manipulação e pré-processamento de dados
- [Scikit-Learn](https://scikit-learn.org/) — Padronização, PCA, validação cruzada, modelos de Machine Learning e Self-Training
- [XGBoost](https://xgboost.readthedocs.io/) — Algoritmos de Gradient Boosting
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — Visualização de dados e gráficos de importância de features

---

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e de pesquisa no contexto do curso de Machine Learning / *Apprentissage Automatique*.
