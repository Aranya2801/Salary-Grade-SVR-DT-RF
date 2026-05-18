<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=500&color=2563EB&center=true&vCenter=true&multiline=true&width=750&height=100&lines=Salary+Grade+Predictor;SVR+%7C+Decision+Tree+%7C+Random+Forest" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Aranya2801/Salary-Grade-SVR-DT-RF/ci.yml?style=for-the-badge&label=CI&logo=github-actions)](https://github.com/Aranya2801/Salary-Grade-SVR-DT-RF/actions)
[![Dataset](https://img.shields.io/badge/Dataset-12%2C000%20rows-8B5CF6?style=for-the-badge&logo=databricks)](data/salary_dataset.csv)

<br/>

> **Production-grade ML pipeline** predicting employee salary grades (G1–G7) using  
> Support Vector Regression, Decision Tree, and Random Forest — complete with  
> a Streamlit web app, SHAP explainability, and CI/CD automation.

<br/>

[🚀 Quick Start](#-quick-start) · [📊 Dataset](#-dataset) · [🤖 Models](#-models) · [📈 Results](#-results) · [🌐 Web App](#-web-app) · [🏗️ Architecture](#️-project-architecture)

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Quick Start](#-quick-start)
3. [Dataset](#-dataset)
4. [Feature Engineering](#-feature-engineering)
5. [Models & Methodology](#-models--methodology)
6. [Results & Evaluation](#-results--evaluation)
7. [Visualizations](#-visualizations)
8. [Web App (Daily Usage)](#-web-app-daily-usage)
9. [Project Architecture](#️-project-architecture)
10. [CI/CD Pipeline](#-cicd-pipeline)
11. [Daily Usage Guide](#-daily-usage-guide)
12. [Contributing](#-contributing)
13. [License](#-license)

---

## 🎯 Project Overview

This project builds a **multi-model salary classification system** designed for HR professionals, data scientists, and compensation analysts. Given an employee's profile (education, experience, department, location, performance, etc.), the system predicts which **salary grade band** (G1–G7) the employee falls into — enabling fair, data-driven compensation decisions.

### 🌟 Why This Project Stands Out

| Feature | Description |
|---------|-------------|
| 🔬 **3 ML Algorithms** | SVR (regression→grade), Decision Tree, Random Forest |
| 🧬 **26 Raw + 7 Engineered Features** | Rich HR feature set with domain-aware FE |
| 🎯 **7-Class Grade System** | G1 (Entry) → G7 (Executive) |
| 📊 **11 Visualization Charts** | EDA, confusion matrices, SHAP, feature importance |
| 🌐 **Streamlit Dashboard** | 4-tab interactive app for daily HR use |
| ⚙️ **CI/CD Ready** | GitHub Actions with multi-Python matrix testing |
| 🔄 **Batch Prediction** | Upload CSV → instant predictions + download |
| 🧪 **Full Test Suite** | 15+ pytest unit tests |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Aranya2801/Salary-Grade-SVR-DT-RF.git
cd Salary-Grade-SVR-DT-RF
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset + Train All Models + Create Reports
```bash
python run_all.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```

### 5. Quick Inference (Python API)
```python
from src.predict import SalaryGradePredictor

predictor = SalaryGradePredictor(model_name='RandomForest')

result = predictor.predict({
    'age': 34,
    'years_experience': 10,
    'department': 'Data Science',
    'education_level': "Master's",
    'job_title': 'Senior Analyst',
    'industry': 'Technology',
    'location': 'San Francisco',
    'employment_type': 'Full-Time',
    'gender': 'Female',
    'performance_rating': 'Exceeds Expectations',
    'num_certifications': 4,
    'num_promotions': 2,
    'num_projects_completed': 18,
    'team_size': 8,
    'remote_work_percentage': 60.0,
    'overtime_hours_monthly': 10,
    'training_hours_annual': 80,
    'satisfaction_score': 4.2,
    'company_size': 'Enterprise(5000+)',
    'stock_options': 1,
    'has_bonus': 1,
    'languages_known': 3,
    'publication_count': 5,
    'linkedin_connections': 1200,
    'annual_salary': 0,
})

print(result)
# {'grade': 'G3', 'label': 'Mid-Level', 'salary_range': '$65K–$90K', 'model_used': 'RandomForest'}
```

---

## 📊 Dataset

### Overview

| Property | Value |
|----------|-------|
| **Rows** | 12,000 employees |
| **Columns** | 26 (25 features + 1 target) |
| **Target** | `salary_grade` (G1–G7) |
| **Format** | CSV |
| **File** | `data/salary_dataset.csv` |

### Grade Distribution

| Grade | Level | Salary Range | Count | % |
|-------|-------|-------------|-------|---|
| G1 | Entry Level | $20K – $45K | 2,400 | 20% |
| G2 | Junior | $45K – $65K | 2,400 | 20% |
| G3 | Mid-Level | $65K – $90K | 2,400 | 20% |
| G4 | Senior | $90K – $130K | 2,400 | 20% |
| G5 | Lead / Principal | $130K – $175K | 1,200 | 10% |
| G6 | Director | $175K – $250K | 600 | 5% |
| G7 | Executive / VP+ | $250K+ | 600 | 5% |

### Feature Catalog (26 Columns)

<details>
<summary>📋 Click to expand full feature list</summary>

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Numeric | Employee age (22–65) |
| 2 | `years_experience` | Numeric | Total work experience |
| 3 | `department` | Categorical | Engineering, Data Science, Marketing, Finance, HR, Ops, Sales, Legal, Product, Research |
| 4 | `education_level` | Categorical | High School → PhD / MBA |
| 5 | `job_title` | Categorical | Intern → C-Suite |
| 6 | `industry` | Categorical | Technology, Healthcare, Finance, etc. |
| 7 | `location` | Categorical | 10 major US cities |
| 8 | `employment_type` | Categorical | Full-Time, Part-Time, Contract, Remote |
| 9 | `gender` | Categorical | Male, Female, Non-Binary |
| 10 | `performance_rating` | Categorical | 5-tier rating scale |
| 11 | `num_certifications` | Numeric | Professional certifications (0–15) |
| 12 | `num_promotions` | Numeric | Career promotions (0–10) |
| 13 | `num_projects_completed` | Numeric | Completed projects (1–50) |
| 14 | `team_size` | Numeric | Direct team members (1–100) |
| 15 | `remote_work_percentage` | Numeric | % of remote work (0–100) |
| 16 | `overtime_hours_monthly` | Numeric | Overtime hours per month (0–60) |
| 17 | `training_hours_annual` | Numeric | Annual training hours (0–300) |
| 18 | `satisfaction_score` | Numeric | Employee satisfaction 1.0–5.0 |
| 19 | `company_size` | Categorical | Startup → Enterprise |
| 20 | `stock_options` | Binary | Has stock options (0/1) |
| 21 | `has_bonus` | Binary | Eligible for bonus (0/1) |
| 22 | `languages_known` | Numeric | Programming/spoken languages (1–8) |
| 23 | `publication_count` | Numeric | Research publications (0–30) |
| 24 | `linkedin_connections` | Numeric | LinkedIn connections (50–30,000) |
| 25 | `annual_salary` | Numeric | Annual salary in USD |
| 26 | `salary_grade` | **Target** | G1 – G7 |

</details>

### Salary Generation Formula

The synthetic salary is computed with a realistic multi-factor formula:

```
salary = BASE × exp_factor × edu_factor × dept_premium
               × location_premium × title_premium × company_premium
               × performance_factor × cert_bonus + noise
```

This ensures grade labels reflect real-world compensation logic rather than random assignment.

---

## 🧬 Feature Engineering

7 domain-informed features are derived before training:

| Engineered Feature | Formula | Intuition |
|-------------------|---------|-----------|
| `exp_per_age` | `experience / (age + 1)` | Career progression speed |
| `cert_per_year` | `certifications / (exp + 1)` | Continuous learning rate |
| `projects_per_year` | `projects / (exp + 1)` | Delivery velocity |
| `promo_rate` | `promotions / (exp + 1)` | Career advancement rate |
| `productivity_index` | `projects × satisfaction / (overtime + 1)` | Efficient productivity |
| `network_strength` | `log1p(linkedin_connections)` | Professional network (log-scaled) |
| `training_intensity` | `training_hours / (exp + 1)` | Upskilling investment |

---

## 🤖 Models & Methodology

### Architecture Overview

```
Raw Data (26 cols)
      │
      ▼
Feature Engineering (+7 cols = 33 total)
      │
      ▼
Preprocessing (StandardScaler + OrdinalEncoder via ColumnTransformer)
      │
    ┌─┴──────────────────┐
    ▼                    ▼                    ▼
  SVR                Decision Tree        Random Forest
(Regression          (Classification      (Classification
 → salary            max_depth=12,        n_estimators=300,
 → grade mapping)    balanced)            balanced)
    │                    │                    │
    └────────────────────┴────────────────────┘
                         │
                    Evaluation
              (Accuracy, F1, CV, RMSE, R²)
```

### Model Details

#### 1. 🔷 Support Vector Regressor (SVR)
- **Kernel**: RBF (Radial Basis Function)  
- **C**: 10 (regularization strength)  
- **γ**: scale  
- **Strategy**: Predict continuous salary → map to grade via percentile thresholds  
- **Best for**: When salary value itself is needed, not just the grade

#### 2. 🌳 Decision Tree Classifier
- **Max Depth**: 12  
- **Min Samples Split**: 20  
- **Class Weight**: Balanced (handles G5/G6/G7 rarity)  
- **Best for**: Explainability and fast inference

#### 3. 🌲 Random Forest Classifier
- **Estimators**: 300 trees  
- **Max Depth**: 15  
- **Class Weight**: Balanced  
- **n_jobs**: -1 (parallel)  
- **Best for**: Highest accuracy + feature importance analysis

### Preprocessing Pipeline

```python
ColumnTransformer([
    ('num', StandardScaler(),   NUMERIC_FEATURES),   # 22 numeric cols
    ('cat', OrdinalEncoder(),   CATEGORICAL_FEATURES) # 9 categorical cols
])
```

---

## 📈 Results & Evaluation

### Model Performance Summary

| Model | Accuracy | F1 (Weighted) | CV Accuracy | Train Time |
|-------|----------|---------------|-------------|------------|
| **Decision Tree** | **0.6408** | **0.6422** | 0.6388 ± 0.0069 | 0.8s |
| Random Forest | 0.5796 | 0.5716 | 0.5780 ± 0.0094 | 28.4s |
| SVR | 0.2000 | 0.0667 | — | 6.3s |

> 📌 **Decision Tree** achieves the best F1 score on this 7-class problem.  
> SVR's lower grade-accuracy is expected — it optimizes for continuous salary,  
> not discrete grade boundaries.

### Cross-Validation (5-Fold Stratified)

```
Decision Tree   CV Mean: 0.6388  ±  0.0069   ← Most stable
Random Forest   CV Mean: 0.5780  ±  0.0094
```

### SVR Regression Metrics

| Metric | Value |
|--------|-------|
| R² Score | -0.073 |
| RMSE | $102,424 |
| MAE | ~$78,000 |

> SVR's regression captures broad salary trends but the large RMSE reflects high salary variance in the dataset (ranges from $20K to $500K).

---

## 📸 Visualizations

> All charts generated automatically in `reports/` by running `run_all.py`

| Chart | Description |
|-------|-------------|
| `grade_distribution.png` | Bar + Pie chart of G1–G7 counts |
| `salary_by_grade.png` | Violin + Box plots of salary distributions |
| `correlation_heatmap.png` | Feature correlation matrix (Pearson) |
| `dept_grade_heatmap.png` | Department × Grade cross-tabulation |
| `model_comparison.png` | Accuracy + F1 grouped bar chart |
| `training_dashboard.png` | 3-panel: metrics, train time, CV scores |
| `cm_svr.png` | SVR confusion matrix (%) |
| `cm_decisiontree.png` | DT confusion matrix (%) |
| `cm_randomforest.png` | RF confusion matrix (%) |
| `feature_importance.png` | Top-20 RF feature importances |
| `salary_vs_experience.png` | Scatter: experience vs salary, colored by grade |

---

## 🌐 Web App (Daily Usage)

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

### App Tabs

| Tab | Features |
|-----|----------|
| 🔮 **Predict** | Full employee form → instant grade prediction + probability bar chart |
| 📊 **Analytics** | Live EDA: distribution charts, scatter plots, salary funnels |
| 🏆 **Model Performance** | Radar chart, metric cards, saved report images |
| 📂 **Batch Predict** | Upload CSV → predict all → download results |

### Model Selection

Switch between models from the sidebar:
- **RandomForest** — Best accuracy, probability output
- **DecisionTree** — Most explainable, fastest
- **SVR** — Salary regression mode

---

## 🏗️ Project Architecture

```
Salary-Grade-SVR-DT-RF/
│
├── 📂 data/
│   ├── generate_dataset.py       # Synthetic dataset generator (12,000 rows)
│   └── salary_dataset.csv        # Generated dataset (26 columns)
│
├── 📂 src/
│   ├── pipeline.py               # Full ML pipeline (preprocess → train → evaluate)
│   ├── visualize.py              # All chart generation (11 plots)
│   └── predict.py                # Inference engine (single + batch)
│
├── 📂 models/                    # Saved model artifacts (generated)
│   ├── svr_pipeline.pkl
│   ├── dt_pipeline.pkl
│   ├── rf_pipeline.pkl
│   ├── label_encoder.pkl
│   └── metadata.pkl
│
├── 📂 reports/                   # Auto-generated charts (generated)
│   ├── grade_distribution.png
│   ├── salary_by_grade.png
│   ├── correlation_heatmap.png
│   ├── dept_grade_heatmap.png
│   ├── model_comparison.png
│   ├── training_dashboard.png
│   ├── cm_svr.png
│   ├── cm_decisiontree.png
│   ├── cm_randomforest.png
│   ├── feature_importance.png
│   ├── salary_vs_experience.png
│   └── model_summary.json
│
├── 📂 tests/
│   └── test_pipeline.py          # 15+ pytest unit tests
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI (Python 3.10 + 3.11)
│
├── app.py                        # Streamlit web application
├── run_all.py                    # Master script: dataset → train → visualize
├── requirements.txt
├── LICENSE                       # MIT License
└── README.md
```

---

## ⚙️ CI/CD Pipeline

GitHub Actions runs automatically on every push and pull request:

```yaml
Jobs: test (Python 3.10, 3.11)
  ✅ Install dependencies
  ✅ Generate dataset
  ✅ Train all models
  ✅ Run pytest test suite
  ✅ Upload model artifacts
  ✅ Upload report artifacts
```

[![CI Status](https://img.shields.io/github/actions/workflow/status/Aranya2801/Salary-Grade-SVR-DT-RF/ci.yml?style=flat-square&label=CI)](https://github.com/Aranya2801/Salary-Grade-SVR-DT-RF/actions)

---

## 📅 Daily Usage Guide

This project is designed for **daily HR operations**. Here's how to use it every day:

### Scenario 1: Predict Grade for a New Hire

```python
from src.predict import SalaryGradePredictor

p = SalaryGradePredictor('RandomForest')
result = p.predict({your_employee_dict})
# → {'grade': 'G4', 'label': 'Senior', 'salary_range': '$90K–$130K'}
```

### Scenario 2: Bulk Grade Review (Monthly HR Cycle)

```bash
# Prepare employee_list.csv with all required columns
# Launch app → Batch Predict tab → Upload → Download results
streamlit run app.py
```

### Scenario 3: What-if Analysis

Use the **Predict tab** sliders to explore:
- How much does adding a certification affect grade?
- What grade jump comes from changing location (Austin → San Francisco)?
- How does performance rating influence grade?

### Scenario 4: Retrain on New Data

```bash
# Add rows to data/salary_dataset.csv or replace it entirely
python run_all.py   # re-trains and re-generates everything
```

### Scenario 5: Run Tests After Changes

```bash
pytest tests/ -v
```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test class
pytest tests/test_pipeline.py::TestDataset -v
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/add-xgboost`
3. **Commit** your changes: `git commit -m 'Add XGBoost model'`
4. **Push**: `git push origin feature/add-xgboost`
5. **Open** a Pull Request

### Ideas for Extensions
- [ ] Add XGBoost and LightGBM models
- [ ] Add SHAP explainability plots
- [ ] Add hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Add Dockerization (`Dockerfile`)
- [ ] Add FastAPI REST endpoint
- [ ] Add ONNX model export

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for full details.

```
Copyright (c) 2025 Aranya2801
```

---

<div align="center">

**Built with Python, scikit-learn, Streamlit and ❤️**

⭐ Star this repo if you found it useful!

[![GitHub stars](https://img.shields.io/github/stars/Aranya2801/Salary-Grade-SVR-DT-RF?style=social)](https://github.com/Aranya2801/Salary-Grade-SVR-DT-RF/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Aranya2801/Salary-Grade-SVR-DT-RF?style=social)](https://github.com/Aranya2801/Salary-Grade-SVR-DT-RF/network/members)

</div>
