import pytest
import pandas as pd
import numpy as np
from ml_explain_preprocess.preprocess import (
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
    filled_age = processed.loc[sample_df['Age'].isnull(), 'Age'].iloc[0]
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

def test_explain_scale_basic(sample_df):
    """Test basic scaling"""
    processed, report = explain_scale(sample_df, columns = ['Age', 'Income'])

    # check report structure
    assert 'before' in report['stats']
    assert 'after' in report['stats']
    assert 'method' in report['stats']

    # for minmax scaling (default), vallues should be btn 0 and 1.
    for col in ['Age', 'Income']:
        for col in processed.columns:
            min_val = processed[col].min()
            max_val = processed[col].max()
            assert min_val >= -0.01     # allow floating point errors
            assert max_val <= 1.01

def test_explain_scale_with_different_methods():
    """Test different scaling methods"""
    df_numeric = pd.DataFrame({
        'Value': [1, 2, 3, 4, 5]
    })

    # test standard scaling
    processed, report = explain_scale(df_numeric, method = 'standard')

    # mean should be close to 0, std close to 1
    assert abs(processed['Value'].mean()) < 0.01
    assert abs(processed['Value'].std() - 1.0) < 0.01

def test_explain_outliers_basic(sample_df_with_outliers):
    """Test outlier detection"""
    processed, report = explain_outliers(sample_df_with_outliers, action = "report")

    # check report structure
    assert "outliers_detected" in report['stats']
    assert "count_affected" in report['stats']

    # should detect outlier in Value1
    assert "Value1" in report['stats']['outliers_detected']

def test_explain_outliers_remove(sample_df_with_outliers):
    """Test outlier removal"""
    original_len = len(sample_df_with_outliers)
    processed, report = explain_outliers(sample_df_with_outliers, action = "remove")

    # should have lesser columns
    assert len(processed) <= original_len

def test_explain_outliers_clip():
    """Test outlier clipping"""
    df_outliers = pd.DataFrame({
        'Value': [1, 2, 3, 4, 100]
    })
    processed, report = explain_outliers(df_outliers, action = "clip")

    # max value should be reduced from 100
    assert processed['Value'].max() < 100

def test_explain_select_features_basic(sample_df):
    """Test feature selection"""
    processed, report = explain_select_features(sample_df)

    # check report structure
    assert "dropped" in report['stats']
    assert "variances" in report['stats']

    # constant column should be dropped
    assert 'Constant' not in processed.columns
    assert 'Constant' in report['stats']['dropped']

def test_explain_select_features_threshold():
    """Tests different thresholds"""
    df_variance = pd.DataFrame({
        'LowVar': [1, 1, 1, 1, 2],       # low var
        'HighVar': [1, 10, 20, 30, 40]     # High var
    })

    # with higher threshold, both might be dropped
    processed, report = explain_select_features(df_variance, threshold = 50)

    # with lower threshold only low variance should be dropped
    processed2, report2 = explain_select_features(df_variance, threshold = 0.1)

    assert 'HighVar' not in processed.columns
    assert 'HighVar' in processed2.columns

def test_explain_preprocess_basic(sample_df):
    """Test full preprocessing pipeline"""
    processed, report = explain_preprocess(sample_df)

    # should return string report
    assert isinstance(report, str)

    # should have some data
    assert processed.shape[0] > 0
    assert processed.shape[1] > 0

    # should not have missing values after preprocesssing
    assert processed.isnull().sum().sum() == 0

def test_explain_preprocess_with_target(sample_df):
    """Test entire preprocessing with target column prottection"""
    processed, report = explain_preprocess(sample_df, target = 'Income')

    # target should be preserved in final df
    assert 'Income' in processed

def test_explain_preprocess_custom_steps():
    """Testing preprocessing with custom steps"""
    processed, report = explain_preprocess(sample_df, steps = ['fill', 'encode'])
    assert isinstance(report, str)
    assert processed.shape[0] > 0

def test_visual_parameter():
    """Test that visual parameter works without errors"""
    df = pd.DataFrame({
        'Numeric': [1, 2, 3, 4, 5],
        'Category': ['A', 'B', 'A', 'B', 'A']
    })

    # testing each function with visual = True (will ensure that no errors arise)
    processed, report = explain_fill_missing(df, visual = True)
    assert isinstance(report, dict)

    processed, report = explain_encode(df, visual = True)
    assert isinstance(report, dict)

    processed, report = explain_scale(df, visual = True)
    assert isinstance(report, dict)

    processed, report = explain_outliers(df, visual = True)
    assert isinstance(report, dict)

    processed, report = explain_select_features(df, visual = True)
    assert isinstance(report, dict)