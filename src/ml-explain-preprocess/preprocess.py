import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from .reports import ExplainReport

def _validate_df(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas dataframe!")
    if df.empty:
        raise ValueError("Dataframe is empty.")
    
def explain_fill_missing(df: pd.DataFrame, strategy: str = 'auto', columns: list = None, visual: bool = False) -> tuple:
    """
    Handles missing values with a detailed & a beginner friendly report.

    Parameters:
    - df: Input dataframe.
    - strategy: 'auto', 'mean', 'median', 'mode', 'constant'.
    - columns: List of columns to process (default -> all with missing values).
    - visual: if True, will generate a missing value heatmap.

    Returns: 
    - processed_df, report_dict
    """
    _validate_df(df)
    df_copy = df.copy()
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()

    report = {
        'explanation': "Missing value handling (imputation) fills in gaps in the data to ensure machine learning models can process it. Missing data can occur due to errors or incomplete collection.", 
        'parameters': f"Strategy: {strategy}, Columns: {columns or 'All with missing'}",
        'stats': {'missing_before': {}, 'missing_after': {}, 'strategies': {}},
        'impact': "",
        'tips': "Choose 'median' for numerical data with outliers, 'mean' for normally distributed data, or 'mode' for categorical data. Always check how much data is missing before imputing."
    }

    total_missing = df.isnull().sum().sum()
    total_cells = df.size()
    percent_missing = (total_missing / total_cells * 100) if total_cells > 0 else 0

    for col in columns:
        missing_before = df_copy[col].isnull().sum()
        report['stats']['missing_before'][col] = f"{missing_before} missing ({missing_before / len(df) * 100:.1f}%)"

        if strategy == 'auto':
            strat = 'median' if pd.api.types.is_numeric_dtype(df_copy[col]) else 'most_frequent'
        else:
            strat = strategy

        imputer = SimpleImputer(strategy = strat, fill_value = 0 if strat == 'constant' else None)
        df_copy[col] = imputer.fit_transform(df_copy[col])

        report['stats']['startegies'][col] = strat
        report['stats']['missing_after'][col] = f"{df_copy[col].isnull().sum()} missing"

    report['impact'] = f"Handled {total_missing} missing values ({percent_missing:.1f}% of data)."

    if visual:
        report['visuals'] = []
        report['visual_descriptions'] = []
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar = False, ax = ax)
        report['visuals'].append('reports/missing_before.png')
        report['visual_descriptions'].append("Heatmap (Before): Red shows missing values, white is non missing.")
        plt.savefig(report['visuals'][-1])
        plt.close()

        fig, ax = plt.subplots()
        sns.heatmap(df_copy.isnull(), cbar = False, ax = ax)
        report['visuals'].append('reports/missing_after.png')
        report['visual_descriptions'].append("Heatmap (After): Should be all white, if all missing values were filled.")
        plt.savefig(report['visuals'][-1])
        plt.close()

    return df_copy, report