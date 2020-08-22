import numpy as np
import pandas as pd


def __add_features__(train_df, test_df):

    concat = pd.concat(
        [
            train_df["anatom_site_general_challenge"],
            test_df["anatom_site_general_challenge"],
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix="site")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat(
        [test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1
    )

    # Sex features
    train_df["sex"] = train_df["sex"].map({"male": 1, "female": 0})
    test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0})
    train_df["sex"] = train_df["sex"].fillna(-1)
    test_df["sex"] = test_df["sex"].fillna(-1)

    # Age features
    train_df["age_approx"] /= train_df["age_approx"].max()
    test_df["age_approx"] /= test_df["age_approx"].max()
    train_df["age_approx"] = train_df["age_approx"].fillna(0)
    test_df["age_approx"] = test_df["age_approx"].fillna(0)

    train_df["patient_id"] = train_df["patient_id"].fillna(0)
    meta_features = ["sex", "age_approx"] + [
        col for col in train_df.columns if "site_" in col
    ]
    meta_features.remove("anatom_site_general_challenge")

    return train_df, test_df, meta_features


def get_features():
    train_df = pd.read_csv(
        "/Users/gsingh/Documents/Personnal/Projects/MelanomaClassification/data/raw/train/train_subset.csv"
    )
    test_df = pd.read_csv(
        "/Users/gsingh/Documents/Personnal/Projects/MelanomaClassification/data/raw/test/test_subset.csv"
    )

    train_df, test_df, meta_features = __add_features__(train_df, test_df)

    return train_df, test_df, meta_features
