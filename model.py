import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from preprocessing import preprocess_data


def train_model():

    df = preprocess_data()

    # -----------------------------------
    # CATEGORICAL COLUMNS
    # -----------------------------------
    categorical_cols = [
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "PriceBand",
        "DurationBucket",
        "RatingTier",
        "Expertise",
        "ExperienceBucket",
    ]

    # Keep only existing columns
    categorical_cols = [
        col for col in categorical_cols if col in df.columns
    ]

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()

        df[col] = df[col].astype(str)

        df[col] = le.fit_transform(df[col])

        label_encoders[col] = le

    # -----------------------------------
    # FEATURES
    # -----------------------------------
    features = [
        "CoursePrice",
        "CourseDuration",
        "CourseRating",
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "PriceBand",
        "DurationBucket",
        "RatingTier",
        "TeacherRating",
        "YearsOfExperience",
    ]

    # Keep only available features
    features = [f for f in features if f in df.columns]

    X = df[features]

    # TARGET
    y = df["Revenue"]

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------------------
    # MODEL
    # -----------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -----------------------------------
    # PREDICTIONS
    # -----------------------------------
    predictions = model.predict(X_test)

    # -----------------------------------
    # EVALUATION
    # -----------------------------------
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 2))

    # -----------------------------------
    # SAVE MODEL
    # -----------------------------------
    joblib.dump(model, "revenue_model.pkl")

    return model, features, df