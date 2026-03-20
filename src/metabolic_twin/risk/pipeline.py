import pandas as pd

from metabolic_twin.features.grandmaster import apply_grandmaster_features

TARGET_COLUMN = "diabetes"
CATEGORICAL_COLUMNS = ("gender", "smoking_history")
CATEGORY_ALIASES = {
    "gender": {
        "female": "Female",
        "male": "Male",
        "other": "Other",
    },
    "smoking_history": {
        "past": "former",
        "no_info": "No Info",
        "no info": "No Info",
        "not_current": "not current",
    },
}
DEFAULT_CATEGORY_FALLBACKS = {
    "gender": "Other",
    "smoking_history": "No Info",
}


def extract_category_levels(raw_df: pd.DataFrame):
    category_levels = {}
    for column in CATEGORICAL_COLUMNS:
        if column in raw_df.columns:
            levels = raw_df[column].dropna().astype(str).unique().tolist()
            category_levels[column] = sorted(levels)
    return category_levels


def normalize_categorical_values(raw_df: pd.DataFrame, category_levels):
    df = raw_df.copy()

    for column, levels in category_levels.items():
        if column not in df.columns:
            continue

        lower_level_map = {str(level).lower(): level for level in levels}
        aliases = CATEGORY_ALIASES.get(column, {})
        fallback = DEFAULT_CATEGORY_FALLBACKS.get(column, levels[0] if levels else None)

        def _normalize(value):
            if pd.isna(value):
                return fallback

            text = str(value).strip()
            if text in levels:
                return text

            lower_text = text.lower()
            if lower_text in lower_level_map:
                return lower_level_map[lower_text]

            aliased_value = aliases.get(lower_text)
            if aliased_value in levels:
                return aliased_value

            return fallback

        df[column] = df[column].map(_normalize)

    return df


def build_risk_feature_frame(raw_df: pd.DataFrame, category_levels=None) -> pd.DataFrame:
    """Apply the production preprocessing used by the risk model."""
    df = raw_df.copy()

    if category_levels:
        df = normalize_categorical_values(df, category_levels)
        for column, levels in category_levels.items():
            if column in df.columns:
                df[column] = pd.Categorical(df[column], categories=levels)

    df_processed = pd.get_dummies(df, drop_first=True)
    return apply_grandmaster_features(df_processed)


def load_risk_training_data(csv_path: str):
    """Load the raw diabetes dataset and return engineered features and labels."""
    raw_df = pd.read_csv(csv_path)
    category_levels = extract_category_levels(raw_df)
    feature_df = build_risk_feature_frame(raw_df, category_levels=category_levels)

    if TARGET_COLUMN not in feature_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training data.")

    X = feature_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = feature_df[TARGET_COLUMN].astype(int)
    return X, y, category_levels


def prepare_risk_inference_features(raw_df: pd.DataFrame, feature_columns, category_levels=None):
    """Apply production preprocessing and align features to the saved training schema."""
    feature_df = build_risk_feature_frame(
        raw_df,
        category_levels=category_levels,
    ).drop(columns=[TARGET_COLUMN], errors="ignore")
    aligned = pd.DataFrame(0.0, index=feature_df.index, columns=list(feature_columns))

    common_columns = [col for col in feature_df.columns if col in aligned.columns]
    if common_columns:
        aligned.loc[:, common_columns] = feature_df[common_columns].astype(float)

    return aligned
