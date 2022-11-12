from sklearn.model_selection import GroupShuffleSplit


def train_test_split_per_column(df, col="uuid", test_size=0.2, n_splits=2, seed=123):
    """Split data into train and test set based on col, rovides randomized train/test
    indices to split data according to a third-party provided group.
    """
    train_inds, test_inds = next(
        GroupShuffleSplit(
            test_size=test_size, n_splits=n_splits, random_state=seed
        ).split(df, groups=df[col])
    )

    return df.iloc[train_inds], df.iloc[test_inds]
