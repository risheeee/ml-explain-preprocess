# ml-explain-preprocess

[![PyPI version](https://badge.fury.io/py/ml-explain-preprocess.svg)](https://badge.fury.io/py/ml-explain-preprocess)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Transparent preprocessing for machine learning with beginner-friendly reports and visualizations.

`ml-explain-preprocess` simplifies data preprocessing for machine learning by providing modular functions to handle missing values, encoding, scaling, outlier detection, and feature selection. Each step generates detailed, easy-to-understand reports (text or JSON) and optional visualizations, making it ideal for beginners and teams needing auditable pipelines.

## Installation

Install from PyPI with:

```bash
pip install ml-explain-preprocess
```

Requires Python 3.8+ and dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn.

## Why ml-explain-preprocess?

Preprocessing is a critical but often opaque step in machine learning. This package makes it transparent by:
- Explaining every transformation to clarify what each step does.
- Generating detailed reports with before/after stats, impacts, and beginner tips.
- Visualizing changes with plots (e.g., missing value heatmaps, histograms).
- Seamless integration with pandas DataFrames, compatible with scikit-learn pipelines.

This package is designed for beginners learning data preparation and teams requiring auditable workflows.

## Key Features

- Missing Value Handling: Impute with mean, median, mode, or constant; reports percentage of missing data handled.
- Encoding: One-hot or label encoding for categorical features, with unique value counts.
- Scaling: Min-max, standard, or robust scaling, with detailed before/after statistics.
- Outlier Detection: IQR or z-score methods, with options to remove, clip, or report outliers.
- Feature Selection: Drop low-variance features, with variance reports.
- Explainable Reports: Beginner-friendly explanations, parameters, impacts, stats, and visual descriptions in text or JSON.
- Visualizations: Save plots (e.g., histograms, boxplots) to a `reports/` folder when `visual=True`.
- Pandas Integration: Input/output as DataFrames for easy use in ML workflows.

## Quickstart

```python
import pandas as pd
from ml_explain_preprocess import explain_preprocess

# Sample dataset
df = pd.DataFrame({
    'Age': [25, 30, None, 40],
    'Gender': ['M', 'F', 'M', 'F'],
    'Income': [50000, 60000, 55000, None]
})

# Run preprocessing pipeline
processed_df, report = explain_preprocess(
    df,
    steps=['fill', 'encode', 'scale'],
    report_format='text',
    visual=True
)

# View report
print(report)
```

Example text report output:

```
Preprocessing Report (Beginner-Friendly)
====================================

Step: FILL
Explanation: Missing value handling (imputation) fills in gaps in the data to ensure machine learning models can process it. Missing data can occur due to errors or incomplete collection.
Parameters Used: Strategy: auto, Columns: ['Age', 'Income']
Impact: Handled 2 missing values (16.7% of data).
Statistics:
  missing_before: {'Age': '1 missing (25.0%)', 'Income': '1 missing (25.0%)'}
  missing_after: {'Age': '0 missing (0%)', 'Income': '0 missing (0%)'}
  strategies: {'Age': 'median', 'Income': 'median'}
Visuals: reports/missing_before.png, reports/missing_after.png
Visual Descriptions:
  - Heatmap (before): Red/yellow shows missing values, white is non-missing.
  - Heatmap (after): Should be all white if all missing values were filled.
Tips for Beginners: Choose 'median' for numerical data with outliers, 'mean' for normally distributed data, or 'mode' for categorical data. Always check how much data is missing before imputing.
----------------------------------------
[... more steps ...]
```

Visualizations (when `visual=True`) are saved to the `reports/` folder.

## Available Functions

- `explain_fill_missing(df, strategy='auto', columns=None, visual=False)`: Impute missing values.
- `explain_encode(df, method='auto', columns=None, visual=False)`: Encode categorical features.
- `explain_scale(df, method='minmax', columns=None, visual=False)`: Scale numerical features.
- `explain_outliers(df, method='iqr', threshold=1.5, action='remove', columns=None, visual=False)`: Handle outliers.
- `explain_select_features(df, threshold=0.01, columns=None, visual=False)`: Drop low-variance features.
- `explain_preprocess(df, steps=['fill', 'encode', 'scale', 'outliers', 'select'], target=None, report_format='json', visual=False)`: Full pipeline with combined report.

Each function returns: `processed_df, report_dict`.

## Project Structure

```
ml-explain-preprocess/
├── ml_explain_preprocess/
│   ├── __init__.py
│   ├── preprocess.py      # Core preprocessing functions
│   └── reports.py         # Report generator
├── tests/                 # Unit tests
├── examples/              # Example Jupyter notebook
├── pyproject.toml         # Package configuration
├── LICENSE                # MIT License
└── README.md
```

## Contributing

We welcome contributions! To get started:
- Fork the repository on [GitHub](https://github.com/risheeee/ml-explain-preprocess).
- Submit issues or pull requests for bugs, features, or improvements.
- Contact: [backpropnomad@gmail.com](mailto:backpropnomad@gmail.com).

## License

Released under the [MIT License](LICENSE).