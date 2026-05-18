# Contributing to Salary Simple Linear Regression

Thank you for considering contributing! This project follows research-grade code standards.

## Development Setup

```bash
git clone https://github.com/Aranya2801/Salary-Simple-Linear-Regression.git
cd Salary-Simple-Linear-Regression
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

All 59 tests must pass before opening a PR.

## Code Style

```bash
black src/ notebooks/ app/ --line-length 100
isort src/ notebooks/ app/
flake8 src/ --max-line-length=100
```

## Pull Request Process

1. Fork → branch (`git checkout -b feature/your-feature`)
2. Write tests for new functionality
3. Ensure all 59 + new tests pass
4. Format with black + isort
5. Update `README.md` if API changes
6. Open PR with clear description

## Reporting Issues

Use GitHub Issues. Include:
- Python version and OS
- Minimal reproducible example
- Full traceback
