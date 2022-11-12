import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoding of features"""

    def __init__(self, incl_cols):
        self.incl_cols = incl_cols
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()

        for col in self.incl_cols:
            X_[col] = self.encoder.fit_transform(X_[col])

        return X_


class ChangeDataTypes(BaseEstimator, TransformerMixin):
    """Changes dtypes for col_dtype_maps"""

    def __init__(self, col_dtype_maps):
        self.col_dtype_maps = col_dtype_maps

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()

        # change dtypes
        for col, dtype in self.col_dtype_maps:
            X_[col] = X_[col].astype(dtype)

        return X_


class HandleNaNValues(BaseEstimator, TransformerMixin):
    """Handle NaN for columns with NaN"""

    def __init__(self, impute=False):
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.impute = impute

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()
        if self.impute:
            for col in X_.columns:
                X_[col] = self.imputer.fit_transform(X_[col].values.reshape(-1, 1))
        else:
            X_ = X_.fillna(0)

        return X_


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scaling of features"""

    def __init__(self, scaler, excl_cols):
        self.scaler = scaler
        self.excl_cols = excl_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()
        X_ = X_.drop(self.excl_cols, axis=1)
        scaled_features = self.scaler.fit_transform(X_)
        scaled_features_df = pd.DataFrame(
            scaled_features, index=X_.index, columns=X_.columns
        )
        unscaled_features = X[self.excl_cols]
        scaled_features_df = pd.concat([scaled_features_df, unscaled_features], axis=1)

        return scaled_features_df


class DropOutliers(BaseEstimator, TransformerMixin):
    """Drop outliers using IQR"""

    def __init__(self, drop=False, thr=3):
        self.drop = drop
        self.thr = thr

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()

        if self.drop:
            Q1 = X_.quantile(0.25)
            Q3 = X_.quantile(0.75)
            IQR = Q3 - Q1
            X_ = X_[~((X_ < (Q1 - 1.5 * IQR)) | (X_ > (Q3 + 1.5 * IQR))).any(axis=1)]
            return X_
        else:
            return X_


class HandleClassImbalance(BaseEstimator, TransformerMixin):
    """Handle class imbalance"""

    def __init__(self, frac, col):
        self.frac = frac
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()

        X_ = X_.drop(X_.loc[X_[self.col] == 0].sample(frac=self.frac).index)
        if len(X_) < 1:
            return X
        else:
            return X_


class OneHotEncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
    """OneHotEncode Categorical Features"""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()

        # binary encode target
        X_ = pd.get_dummies(data=X_, columns=self.features)

        return X_


class OrdinalEncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
    """Ordinal Categorical Features"""

    def __init__(self, features):
        self.features = features
        self.oe = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # copy original DataFrame
        X_ = X.copy()

        # ordinal encode target
        for col in self.features:
            X_[col] = X_[col] = self.oe.fit_transform(X_[col].values.reshape(-1, 1))

        return X_
