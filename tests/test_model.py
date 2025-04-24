import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from football_predictions import compute_team_statistics, load_data

def test_model_accuracy():
    """
    Test that the model achieves at least 50% accuracy.
    This is a sanity check to ensure the model is learning something.
    """
    # Load and process data
    df = load_data()
    stats_df = compute_team_statistics(df)

    # Define features and target
    features = ['Goals Scored', 'Goals Conceded', 'Goal Difference',
                'Win Rate', 'xG Scored', 'xG Conceded', 'xG Difference']
    X = stats_df[features]
    y = (stats_df['Win Rate'] > 0.5).astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)

    # Assert accuracy threshold
    assert acc > 0.5, f"Expected accuracy > 0.5, but got {acc:.2f}"
