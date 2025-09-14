import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from .reports import ExplainReport
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

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
    total_cells = df.size
    percent_missing = (total_missing / total_cells * 100) if total_cells > 0 else 0

    for col in columns:
        missing_before = df_copy[col].isnull().sum()
        report['stats']['missing_before'][col] = f"{missing_before} missing ({missing_before / len(df) * 100:.1f}%)"

        if strategy == 'auto':
            strat = 'median' if pd.api.types.is_numeric_dtype(df_copy[col]) else 'most_frequent'
        else:
            strat = strategy

        imputer = SimpleImputer(strategy = strat, fill_value = 0 if strat == 'constant' else None)
        df_copy[col] = imputer.fit_transform(df_copy[[col]])

        report['stats']['strategies'][col] = strat
        report['stats']['missing_after'][col] = f"{df_copy[col].isnull().sum()} missing"

    report['impact'] = f"Handled {total_missing} missing values ({percent_missing:.1f}% of data)."

    if visual:
        os.makedirs('reports', exist_ok=True)

        report['visuals'] = []
        report['visual_descriptions'] = []

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=True, ax=ax, cmap='viridis')
            ax.set_title('Missing Values - Before')
            report['visuals'].append('reports/missing_before.png')
            report['visual_descriptions'].append("Heatmap (Before): Yellow shows missing values, dark purple is non missing.")
            plt.savefig(report['visuals'][-1], dpi=150, bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_copy.isnull(), cbar=True, ax=ax, cmap='viridis')
            ax.set_title('Missing Values - After')
            report['visuals'].append('reports/missing_after.png')
            report['visual_descriptions'].append("Heatmap (After): Should be all dark purple if all missing values were filled.")
            plt.savefig(report['visuals'][-1], dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            report['visual_descriptions'].append(f"Could not generate heatmap: {str(e)}")

    return df_copy, report

def explain_encode(df: pd.DataFrame, method: str = 'auto', columns: list = None, visual: bool = False) -> tuple:
    """
    Encodes categorical features with a detailed and a beginner friendly report.

    Parameters:
    - df: Input Dataframe.
    - method: 'auto', 'onehot', 'label'.
    - columns: List of categorical columns (default: auto detect).
    - visual: If True, plot unique values before / after.

    Returns:
    - processed_df, report_dict.
    """
    _validate_df(df)
    df_copy = df.copy()
    if columns is None:
        columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

    report = {
        'explanation': "Encoding converts categorical data (e.g., 'Male'/'Female') into numbers so machine learning models can understand it. One-hot encoding creates new columns for each category, while label encoding assigns numbers to categories.",
        'parameters': f"Method: {method}, Columns: {columns or 'Auto detect categoricals'}",
        'stats': {'unique_before': {}, 'unique_after': {}, 'methods': {}, 'new_columns': {}},
        'impact': "",
        'tips': "Use one-hot encoding for low-cardinality categories (e.g., <10 unique values) to avoid implying order. Label encoding is better for high-cardinality or ordinal data."
    }

    original_cols = df_copy.columns.tolist()

    for col in columns:
        unique_before = df_copy[col].nunique()
        report['stats']['unique_before'][col] = f"{unique_before} unique values"

        if method == 'auto':
            meth = 'onehot' if unique_before < 10 else 'label'
        else:
            meth = method

        if meth == 'onehot':
            encoder = OneHotEncoder(sparse_output = False, drop = 'first')
            encoded = encoder.fit_transform(df_copy[[col]])
            encoded_df = pd.DataFrame(encoded, columns = encoder.get_feature_names_out(), index = df_copy.index)
            df_copy = pd.concat([df_copy.drop(col, axis=1), encoded_df], axis=1)
            report['stats']['methods'][col] = 'onehot'
            report['stats']['new_columns'][col] = encoder.get_feature_names_out().tolist()
        else:
            encoder = LabelEncoder()
            df_copy[col] = encoder.fit_transform(df_copy[col])
            report['stats']['methods'][col] = 'label'

        report['stats']['unique_after'][col] = "See new columns" if meth == 'onehot' else f"{df_copy[col].nunique()} unique values"

    new_cols = len(df_copy.columns) - len(original_cols)
    report['impact'] = f"Transformed {len(columns)} categorical columns, added {new_cols} new columns."

    if visual:
        os.makedirs('reports', exist_ok=True)
        
        report['visuals'] = []
        report['visual_descriptions'] = []
        
        try:
            # Check if there are any categorical columns to plot
            if report['stats']['unique_before']: 
                unique_counts = {}
                for k, v in report['stats']['unique_before'].items():
                    try:
                        count = int(v.split()[0])
                        unique_counts[k] = count
                    except (ValueError, IndexError):
                        continue
                
                if unique_counts:  # Only plot if we have data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pd.Series(unique_counts).plot(kind='bar', ax=ax)
                    ax.set_title("Unique Values Before Encoding")
                    ax.set_xlabel("Categorical Columns")
                    ax.set_ylabel("Number of Unique Values")
                    plt.xticks(rotation=45)
                    report['visuals'].append('reports/encode_uniques.png')
                    report['visual_descriptions'].append("Bar plot: Shows number of unique values per categorical column before encoding.")
                    plt.savefig(report['visuals'][-1], dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    report['visual_descriptions'].append("No categorical columns found - no bar plot generated.")
            else:
                report['visual_descriptions'].append("No categorical columns to encode - no visualization generated.")
                
        except Exception as e:
            report['visual_descriptions'].append(f"Could not generate encoding plot: {str(e)}")

    return df_copy, report

def explain_scale(df: pd.DataFrame, method: str = 'minmax', columns: list = None, visual: bool = False) -> tuple:
    """
    Scale numerical features with a detailed & beginner friendly report.

    Parameters:
    - df: Input Dataframe
    - method: 'minmax', 'standard', 'robust'.
    - columns: List of numerical columns (default: auto detect).
    - visual: If true, histograms before / after.

    Returns:
    - proccesed_df, report_dict
    """
    _validate_df(df)
    df_copy = df.copy()
    if columns is None:
        columns = df.select_dtypes(include = ['int', 'float']).columns.tolist()

    report = {
        'explanation': "Scaling adjusts numerical features to a common range (e.g., 0 to 1) so that machine learning models treat all features equally. MinMax scales to [0,1], Standard scales to mean=0, std=1, Robust is outlier-resistant.",
        'parameters': f"Method: {method}, Columns: {columns or 'auto detect numericals'}",
        'stats': {'before': {}, 'after': {}, 'method': {}},
        'impact': "",
        'tips': "Use scaling for algorithms sensitive to feature magnitudes (e.g., SVM, KNN, neural networks). MinMax is good for bounded ranges, Standard for normally distributed data."
    }

    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler(),
    }
    scaler = scalers.get(method, MinMaxScaler())

    for col in columns:
        stats_before = {
            'min': df_copy[col].min(),
            'max': df_copy[col].max(),
            'mean': df_copy[col].mean(),
            'std': df_copy[col].std(),
        }
        report['stats']['before'][col] = f"Min: {stats_before['min']:.2f}, Max: {stats_before['max']:.2f}, Mean: {stats_before['mean']:.2f}, Std: {stats_before['std']:.2f}"

        df_copy[col] = scaler.fit_transform(df_copy[[col]])

        stats_after = {
            'min': df_copy[col].min(),
            'max': df_copy[col].max(),
            'mean': df_copy[col].mean(),
            'std': df_copy[col].std(),
        }

        report['stats']['after'][col] = f"Min: {stats_after['min']:.2f}, Max: {stats_after['max']:.2f}, Mean: {stats_after['mean']:.2f}, Std: {stats_after['std']:.2f}"

        report['stats']['method'][col] = method

    report['impact'] = f"Scaled {len(columns)} numerical columns to {method} range."

    if visual:
        os.makedirs('reports', exist_ok=True)
        
        report['visuals'] = []
        report['visual_descriptions'] = []
        
        for col in columns:
            try:
                # Check if column has valid numeric data
                if df[col].notna().sum() > 0 and df_copy[col].notna().sum() > 0:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Plot before scaling
                    df[col].hist(ax=axs[0], bins=20, alpha=0.7)
                    axs[0].set_title(f'{col} Before Scaling')
                    axs[0].set_xlabel('Value')
                    axs[0].set_ylabel('Frequency')
                    
                    # Plot after scaling
                    df_copy[col].hist(ax=axs[1], bins=20, alpha=0.7)
                    axs[1].set_title(f'{col} After Scaling')
                    axs[1].set_xlabel('Value')
                    axs[1].set_ylabel('Frequency')

                    plt.tight_layout()
                    clean_col = re.sub(r'[^\w\-_.]', '_', col)
                    filename = f'reports/scale_{clean_col}.png'
                    
                    report['visuals'].append(filename)
                    report['visual_descriptions'].append(f"Histograms for {col}: Left shows distribution before scaling, right shows after {method} scaling.")
                    
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    report['visual_descriptions'].append(f"Skipped histogram for {col}: No valid data to plot.")
                    
            except Exception as e:
                report['visual_descriptions'].append(f"Could not generate histogram for {col}: {str(e)}")
                plt.close('all')

    return df_copy, report

def explain_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5, action: str = 'remove', columns: list = None, visual: bool = False) -> tuple:
    """
    Detect and handle outliers with a detailed and beginner friendly report.

    Parameters:
    - df: Input datafram.
    - method: 'iqr' or 'zscore'.
    - threshold: For IQR (1.5) or z-score (3).
    - action: 'remove', 'clip', 'report'.
    - columns: Numerical columns.
    - visual: Boxplots before / after.

    Returns:
    - processed_df, report_dict 
    """
    _validate_df(df)
    df_copy = df.copy()
    if columns is None:
        columns = df.select_dtypes(include = ['int', 'float']).columns.tolist()

    report = {
        'explanation': "Outliers are extreme values that differ significantly from most data points. They can skew model training. IQR uses quartiles to detect outliers, z-score uses standard deviations.",
        'parameters': f"Method: {method}, Threshold: {threshold}, Action: {action}, Columns: {columns or 'Auto detected numerics'}",
        'stats': {'outliers_detected': {}, 'count_affected': {}},
        'impact': {},
        'tips': "Remove outliers for small datasets or sensitive models (e.g., linear regression). Clip outliers to keep data but reduce their impact. Always inspect outliers to ensure they're not meaningful."
    }

    total_outliers = 0
    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = ((df_copy[col] < lower) | (df_copy[col] > upper))
        else:   # zscore
            z = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
            outliers = z > threshold

        count_outliers = outliers.sum()
        total_outliers += count_outliers
        report['stats']['outliers_detected'][col] = f"{count_outliers} outliers ({count_outliers / len(df) * 100:.1f}%)"

        if action == 'remove':
            df_copy = df_copy[~outliers]
            report['stats']['count_affected'][col] = f"Removed {count_outliers} rows"
        elif action == 'clip':
            df_copy[col] = np.clip(df_copy[col], lower, upper) if method == 'iqr' else np.clip(df_copy[col], df_copy[col].mean() - threshold * df_copy[col].std(), df_copy[col].mean() + threshold * df_copy[col].std())
            report['stats']['count_affected'][col] = f"Clipped {count_outliers} values"
        else:
            report['stats']['count_affected'][col] = "Reported, no changes"

    report['impact'] = f"Processed {total_outliers} outliers accross {len(columns)} columns."

    if visual:
        os.makedirs('reports', exist_ok=True)  # Fix: Create directory
        
        report['visuals'] = []
        report['visual_descriptions'] = []
        
        for col in columns:
            try:
                # Check if column has valid numeric data
                if df[col].notna().sum() > 1: 
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Plot before outlier handling
                    sns.boxplot(y=df[col], ax=axs[0])
                    axs[0].set_title(f'{col} Before')
                    axs[0].set_ylabel('Value')
                    
                    # Plot after outlier handling
                    if action == 'remove' and len(df_copy) != len(df):
                        if method == 'iqr':
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - threshold * IQR
                            upper = Q3 + threshold * IQR
                            non_outliers = df[col][(df[col] >= lower) & (df[col] <= upper)]
                        else:  # zscore
                            z = np.abs((df[col] - df[col].mean()) / df[col].std())
                            non_outliers = df[col][z <= threshold]
                        
                        sns.boxplot(y=non_outliers, ax=axs[1])
                        axs[1].set_title(f'{col} After (Outliers Removed)')
                    else:
                        # For clipping or reporting: show the modified data
                        sns.boxplot(y=df_copy[col], ax=axs[1])
                        if action == 'clip':
                            axs[1].set_title(f'{col} After (Outliers Clipped)')
                        else:
                            axs[1].set_title(f'{col} After')
                    
                    axs[1].set_ylabel('Value')
                    plt.tight_layout()

                    clean_col = re.sub(r'[^\w\-_.]', '_', col)
                    filename = f'reports/outliers_{clean_col}.png'
                    
                    report['visuals'].append(filename)
                    
                    if action == 'remove':
                        description = f"Boxplots for {col}: Left shows original data with outliers, right shows data that would remain after removing outliers."
                    elif action == 'clip':
                        description = f"Boxplots for {col}: Left shows original data, right shows data after clipping outliers to threshold boundaries."
                    else:
                        description = f"Boxplots for {col}: Left shows original data with outliers as points beyond whiskers, right shows the same data (report only)."
                    
                    report['visual_descriptions'].append(description)
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                else:
                    report['visual_descriptions'].append(f"Skipped boxplot for {col}: Not enough valid data points.")
                    
            except Exception as e:
                report['visual_descriptions'].append(f"Could not generate boxplot for {col}: {str(e)}")
                plt.close('all')

    return df_copy, report

def explain_select_features(df: pd.DataFrame, threshold: float = 0.01, columns: list = None, visual: bool = False) -> tuple:
    """
    Select features by variance with a detailed and a beginner ffriendly report.

    Parameters:
    - df: Input Dataframe.
    - threshold: Variance threshold.
    - columns: Columns to check.
    - visual: plot variances.

    Returns:
    - processed_df, report_dict
    """
    _validate_df(df)
    df_copy = df.copy()
    if columns is None:
        columns = df.select_dtypes(include = ['number']).columns.tolist()

    report = {
        'explanation': "Feature selection removes columns with low variance (i.e., nearly constant values) because they provide little information for models. Variance measures how spread out data is.",
        'parameters': f"Variance Threshold: {threshold}, Columns: {columns or 'Auto detected numerics'}",
        'stats': {'columns_before': list(df.columns), 'dropped': [], 'variances': {}},
        'impact': "",
        'tips': "Use a low threshold to keep more features, higher to be stricter. Check dropped columns to ensure no important data is lost."
    }

    for col in columns:
        var = df_copy[col].var()
        report['stats']['variances'][col] = f"Variance: {var:.4f}"
        if var < threshold:
            report['stats']['dropped'].append(col)

    selector = VarianceThreshold(threshold = threshold)
    numeric_df = df_copy[columns]
    
    try:
        selected = selector.fit_transform(numeric_df)
        selected_columns = numeric_df.columns[selector.get_support()].tolist()
        
        non_numeric_df = df_copy.drop(columns, axis = 1)
        if selected_columns:  # If any columns remain after selection
            df_copy = pd.concat([non_numeric_df, df_copy[selected_columns]], axis = 1)
        else:  # If all numeric columns were dropped
            df_copy = non_numeric_df
            
    except ValueError as e:
        # Handle case where no features meet the threshold
        if "No feature in X meets the variance threshold" in str(e):
            # All numeric columns will be dropped
            selected_columns = []
            df_copy = df_copy.drop(columns, axis = 1)
        else:
            raise e

    report['stats']['columns_after'] = list(df_copy.columns)
    report['impact'] = f"Dropped {len(report['stats']['dropped'])} low variance columns."

    if visual:
        os.makedirs('reports', exist_ok=True) 
        
        report['visuals'] = []
        report['visual_descriptions'] = []
        
        try:
            # Check if there are any variances to plot
            if report['stats']['variances']:
                variance_data = {}
                for k, v in report['stats']['variances'].items():
                    try:
                        # Extract variance 
                        if ': ' in v:
                            variance_val = float(v.split(': ')[1])
                            variance_data[k] = variance_val
                        else:
                            variance_data[k] = float(v)
                    except (ValueError, IndexError):
                        continue
                
                if variance_data:  # Only plot if we have valid data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create horizontal bar plot
                    variance_series = pd.Series(variance_data).sort_values()
                    variance_series.plot(kind='barh', ax=ax)
                    
                    # Add threshold line
                    ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

                    ax.set_title('Feature Variances', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Variance', fontsize=12)
                    ax.set_ylabel('Features', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Color bars based on whether they meet threshold
                    bars = ax.patches
                    for i, (feature, var_val) in enumerate(variance_series.items()):
                        if var_val < threshold:
                            bars[i].set_color('red')
                            bars[i].set_alpha(0.7)
                        else:
                            bars[i].set_color('green')
                            bars[i].set_alpha(0.7)
                    
                    plt.tight_layout()
                    
                    report['visuals'].append('reports/variances.png')
                    
                    dropped_count = len([v for v in variance_data.values() if v < threshold])
                    kept_count = len(variance_data) - dropped_count
                    
                    description = f"Bar plot: Shows variance of each numerical column. Red dashed line is the threshold ({threshold}). Red bars (low variance) will be dropped ({dropped_count} features), green bars (high variance) will be kept ({kept_count} features)."
                    report['visual_descriptions'].append(description)
                    
                    plt.savefig(report['visuals'][-1], dpi=150, bbox_inches='tight')
                    plt.close()
                    
                else:
                    report['visual_descriptions'].append("No valid variance data found - no bar plot generated.")
            else:
                report['visual_descriptions'].append("No numerical columns to analyze - no variance plot generated.")
                
        except Exception as e:
            report['visual_descriptions'].append(f"Could not generate variance plot: {str(e)}")
            plt.close('all')

    return df_copy, report

def explain_preprocess(df: pd.DataFrame, steps: list = ['fill', 'encode', 'scale', 'outliers', 'select'], target: str = None, report_format: str = 'json', visual: bool = False) -> tuple:
    """
    Full preprocessing pipeline with a combined, detailed and a beginner friendly report.

    Parameters:
    - df: Input Dataframe.
    - steps: List of steps to apply (in order).
    - target: Target Column.
    - report_format: 'json' or 'text'.
    - visual: if True, generates visual plots for all steps.

    Returns:
    - processed_df, combined_report
    """
    _validate_df(df)
    df_copy = df.copy()
    if target and target in df_copy.columns:
        target_col = df_copy.pop(target)        # protect target col

    report = ExplainReport()

    func_map = {
        'fill': explain_fill_missing,
        'encode': explain_encode,
        'scale': explain_scale,
        'outliers': explain_outliers,
        'select': explain_select_features,
    } 

    for step in steps:
        if step not in func_map:
            raise ValueError(f"Invalid step: {step}")
        df_copy, step_report = func_map[step](df_copy, visual = visual)
        report.add_step(step, step_report)

    if target and target in target_col.name:
        df_copy[target] = target_col

    combined = report.generate(format = report_format)
    return df_copy, combined