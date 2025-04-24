import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = "data/"  # adjust this if your CSVs are elsewhere


def load_data():
    """Loads raw datasets from CSV files."""
    regular_season = pd.read_csv(f"{DATA_DIR}NSL_regular_season_data_2.csv")
    return regular_season


def compute_team_statistics(df):
    """Computes team-level statistics for the regular season."""
    df['Winner'] = df.apply(
        lambda x: x['HomeTeam'] if x['HomeScore'] > x['AwayScore'] else (
            x['AwayTeam'] if x['HomeScore'] < x['AwayScore'] else 'Draw'),
        axis=1
    )

    stats = {
        'Wins': df.groupby('Winner').size(),
        'Goals Scored': df.groupby('HomeTeam')['HomeScore'].sum() +
                        df.groupby('AwayTeam')['AwayScore'].sum(),
        'Goals Conceded': df.groupby('HomeTeam')['AwayScore'].sum() +
                          df.groupby('AwayTeam')['HomeScore'].sum(),
    }

    stats_df = pd.DataFrame(stats).fillna(0)
    stats_df['Goal Difference'] = stats_df['Goals Scored'] - stats_df['Goals Conceded']
    stats_df = stats_df.drop('Draw', errors='ignore')

    total_games = df.groupby('HomeTeam').size() + df.groupby('AwayTeam').size()
    stats_df['Win Rate'] = stats_df['Wins'] / total_games

    stats_df['xG Scored'] = df.groupby('HomeTeam')['Home_xG'].sum() + \
                            df.groupby('AwayTeam')['Away_xG'].sum()
    stats_df['xG Conceded'] = df.groupby('HomeTeam')['Away_xG'].sum() + \
                              df.groupby('AwayTeam')['Home_xG'].sum()
    stats_df['xG Difference'] = stats_df['xG Scored'] - stats_df['xG Conceded']

    return stats_df.dropna()


def train_model(X, y):
    """Trains and evaluates a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("Predictions:", y_pred[:10])  # Show first few predictions

    return model


def main():
    df = load_data()
    stats_df = compute_team_statistics(df)

    # Select features and target
    features = ['Goals Scored', 'Goals Conceded', 'Goal Difference',
                'Win Rate', 'xG Scored', 'xG Conceded', 'xG Difference']
    X = stats_df[features]
    y = (stats_df['Win Rate'] > 0.5).astype(int)  # Example binary target

    model = train_model(X, y)


if __name__ == "__main__":
    main()
