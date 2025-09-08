import pytest
import pandas as pd
import numpy as np
from ml_explain_preprocess import (
    explain_fill_missing,
    explain_encode,
    explain_scale,
    explain_outliers,
    explain_select_features,
    explain_preprocess
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Age': [25, 30, None, 40, 35],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Income': [50000, 60000, 55000, None, 70000],
        'Constant': [1, 1, 1, 1, 1],
        'Score': [85.5, 90.0, 78.5, 95.0, 88.0]
    })

@pytest.fixture
def sample_df_with_outliers():
    return pd.DataFrame({
        'Value1': [1, 2, 3, 4, 100],     # 100 is an outlier
        'Value2': [10, 15, 12, 14, 13],
        'Category': ['A', 'B', 'A', 'C', 'B']
    })

def test_explain_fill_missing_basic(sample_df):
    """Tests basic missing value handling"""
    processed, report = explain_fill_missing(sample_df)

    assert processed.isnull().sum().sum() == 0      # check that missing values are filled

    # check report structure
    assert isinstance(report, dict)
    assert 'explanation' in report
    assert 'stats' in report
    assert 'strategies' in report['stats']
    assert 'missing_before' in report['stats']
    assert 'missing_after' in report['stats']

    # check that strategies were applied
    assert len(report['stats']['strategies']) > 0

def test_explain_fill_missing_with_strategy(sample_df):
    """Test specific strategy"""
    processed, report = explain_fill_missing(sample_df, strategy = 'median')

    # age should be filled with median
    original_median = sample_df['Age'].median()
    filled_age = processed.iloc[sample_df['Age'].isnull(), 'Age'].iloc[0]
    assert filled_age == original_median

def test_explain_fill_missing_empty_columns():
    """Test when no columns have missing values"""
    df_no_missing = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    processed, report = explain_fill_missing(df_no_missing)

    # it should return original df
    pd.testing.assert_frame_equal(processed, df_no_missing)
    assert report['stats']['missing_before'] == {}

def test_explain_encode_basic(sample_df):
    """Test basic encoding"""
    processed, report = explain_encode(sample_df)

    # check that categorical columns are processed
    assert isinstance(report, dict)
    assert 'methods' in report['stats']

    # Gender should be encoded (onehot or label)
    if 'Gender' in report['stats']['methods']:
        if report['stats']['methods']['Gender'] == 'onehot':
            # should have columns of onehot
            assert any('Gender_' in col for col in processed.columns)
        else:
            # should be numeric for label encoding
            assert pd.api.types.is_numeric_dtype(processed['Gender'])

def test_explain_encode_onehot():
    """Test one hot encoding"""
    df_cat = pd.DataFrame({
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue']
    })
    processed, report = explain_encode(df_cat, method = 'onehot')

    # should create new columns
    assert len(processed.columns > 1)
    assert report['stats']['methods']['Color'] == 'onehot'

def test_explain_encode_label():
    """Test label encoding"""
    df_cat = pd.DataFrame({
        'Size': ['Small', 'Medium', 'Large', 'Small']
    })
    processed, report = explain_encode(df_cat, method = 'label')

    # should be numeric
    assert pd.api.types.is_numeric_dtype(processed['Size'])
    assert report['stats']['methods']['Size'] == 'label'

