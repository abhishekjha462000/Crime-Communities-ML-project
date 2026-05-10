"""
Custom transformer classes for the Communities & Crime pipeline.
Required for unpickling the saved joblib models.

Drop this file into the root of the Streamlit project directory.
The Streamlit app should import from this file BEFORE calling joblib.load().
"""

from sklearn.base import BaseEstimator, TransformerMixin


class LemasIndicatorAdder(BaseEstimator, TransformerMixin):
    """Adds a binary `lemas_observed` column based on whether reference_col is missing."""

    def __init__(self, reference_col='LemasSwornFT'):
        self.reference_col = reference_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['lemas_observed'] = (~X[self.reference_col].isna()).astype(int)
        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns if present (used by the no-race sensitivity pipeline)."""

    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_present = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols_present) if cols_present else X.copy()
