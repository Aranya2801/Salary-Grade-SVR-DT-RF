# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2025

### Added
- `SalaryRegressionEngine` — full OLS inference engine
- Bootstrap confidence intervals (1,000 iterations, BCa)
- Prediction intervals vs confidence intervals (separate methods)
- 6-panel diagnostic dashboard (Shapiro-Wilk, DW, BP, Cook's D, Q-Q, Scale-Location)
- 5-fold cross-validation (R², RMSE, MAE)
- `salary_dataset.csv` — 500 rows, 10 features
- 59 unit tests across 8 test classes (100% passing)
- GitHub Actions CI: 3 OS × 4 Python versions
- Interactive Streamlit salary predictor app
- `pyproject.toml` packaging configuration
- MIT License

### Changed
- Replaced single Excel file with full modular Python package
- README completely rewritten to research-lab standard

---

## [1.0.0] — 2024

### Added
- Initial Excel-based salary regression dataset
- Basic README placeholder
