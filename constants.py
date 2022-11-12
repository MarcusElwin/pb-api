from features.features import (
    BinaryLabelEncoder,
    HandleNaNValues,
    ChangeDataTypes,
    OrdinalEncodeCategoricalFeatures,
    FeatureScaler,
)
from sklearn.preprocessing import StandardScaler

CATEGORICAL_FEATURES = ["merchant_category", "merchant_group", "name_in_email"]

EXCL_FEATURES = []

TARGET = "default"
USER_COL = "uuid"

COL_DTYPE_MAPS = [
    ("account_status", "int64"),
    ("account_worst_status_0_3m", "int64"),
    ("account_worst_status_12_24m", "int64"),
    ("account_worst_status_3_6m", "int64"),
    ("account_worst_status_6_12m", "int64"),
    ("num_arch_written_off_0_12m", "int64"),
    ("num_arch_written_off_12_24m", "int64"),
    ("worst_status_active_inv", "int64"),
]

PREP_STEPS = [
    ("binary_encoding", BinaryLabelEncoder(incl_cols=["has_paid"])),
    ("handle_nan", HandleNaNValues(impute=False)),
    ("change_dtypes", ChangeDataTypes(col_dtype_maps=COL_DTYPE_MAPS)),
]
FEATURE_STEPS = [
    (
        "ordinal_encode_cat",
        OrdinalEncodeCategoricalFeatures(features=CATEGORICAL_FEATURES),
    ),
    (
        "std_scale_data",
        FeatureScaler(StandardScaler(), excl_cols=CATEGORICAL_FEATURES),
    ),
]
