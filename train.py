import click, logging
import pandas as pd
from joblib import dump
from features.utils import train_test_split_per_column
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from constants import (
    CATEGORICAL_FEATURES,
    COL_DTYPE_MAPS,
    EXCL_FEATURES,
    TARGET,
    USER_COL,
    FEATURE_STEPS,
    PREP_STEPS,
)
from typing import Dict, Tuple, Union
from pathlib import Path

# set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def get_training_pipeline(class_weight: Dict[int, int] = "balanced") -> Pipeline:
    model_steps = [
        (
            "random_forest_clf",
            RandomForestClassifier(
                random_state=123, n_estimators=2000, n_jobs=-1, class_weight=class_weight
            ),
        )
    ]
    return Pipeline(PREP_STEPS + FEATURE_STEPS + model_steps)


def evaluate_model_cv(
    clf: Union[RandomForestClassifier],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    num_folds: int = 5,
):
    scv = StratifiedKFold(n_splits=num_folds)
    metric_names = ["roc_auc"]
    scores_df = pd.DataFrame(index=metric_names, columns=["Stratified-CV"])
    for metric in metric_names:
        logging.info(f"Starting CV for metric: {metric}")
        score = cross_val_score(clf, X_train, y_train, scoring=metric, cv=scv).mean()
        scores_df.loc[metric] = [score]

    print(scores_df)


def get_training_data(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split_per_column(dataset, col=USER_COL)

    X_train, X_test = df_train.drop([TARGET, USER_COL], axis=1), df_test.drop(
        [TARGET, USER_COL], axis=1
    )
    y_train, y_test = df_train[TARGET], df_test[TARGET]
    return X_train, y_train, X_test, y_test


@click.command()  # add click cli instead of argparse
@click.option(
    "--train-data-path",
    default="../dataset.csv",
    show_default=True,
    help="Path to training data",
    type=str,
)
@click.option(
    "--save-model-path",
    default="/tmp/",
    show_default=True,
    help="Path where to save model",
    type=str,
)
def main(train_data_path: str, save_model_path: str) -> None:
    """Main script for running model training"""
    logging.info(f"Loading data from {str(Path(train_data_path))}")
    if Path(train_data_path).exists():
        dataset = pd.read_csv(Path(train_data_path), sep=";")
    else:
        raise FileExistsError(f"No file found at this path: {train_data_path}...")

    logging.info(f"Inspect data...")
    print(f"Number of rows {len(dataset)}")
    print(f"Number of uuid {dataset['uuid'].nunique()}")

    if len(EXCL_FEATURES) > 0:
        dataset = dataset.drop(EXCL_FEATURES, axis=1)

    # exclude prediction data
    pred_dataset = dataset[dataset[TARGET].isna()]
    train_dataset = dataset[~dataset[TARGET].isna()]

    logging.info("Get training data...")
    X_train, y_train, X_test, y_test = get_training_data(train_dataset)

    model_pipeline = get_training_pipeline()

    logging.info(f"Start training...")
    model = model_pipeline.fit(X_train, y_train)

    logging.info("Evaluate model...")
    evaluate_model_cv(model, X_train, y_train, num_folds=5)

    logging.info(f"Make prediction on new data...")
    input = pred_dataset.drop([TARGET, USER_COL], axis=1)
    pred_dataset = pred_dataset[[USER_COL, TARGET]]
    pred_dataset[TARGET] = model.predict_proba(input)[:, 1]  # # only positive class

    logging.info(f"Saving predictions...")
    pred_dataset.to_csv(Path(save_model_path) / "predictions.csv")

    logging.info(f"Save model...")
    dump(model, Path(save_model_path) / "model.joblib")


if __name__ == "__main__":
    main()
