"""
General utility functions for NCAA March Madness prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== SUBMISSION HELPERS ====================

def parse_submission_id(submission_id: str) -> Tuple[int, int, int]:
    """
    Parse submission ID into components.

    Args:
        submission_id: String in format 'SSSS_XXXX_YYYY'

    Returns:
        Tuple of (season, team1_id, team2_id)
    """
    parts = submission_id.split('_')
    return int(parts[0]), int(parts[1]), int(parts[2])


def create_submission_id(season: int, team1: int, team2: int) -> str:
    """
    Create submission ID from components.

    Args:
        season: Season year
        team1: First team ID (lower ID)
        team2: Second team ID (higher ID)

    Returns:
        Submission ID string
    """
    # Ensure team1 < team2
    if team1 > team2:
        team1, team2 = team2, team1

    return f"{season}_{team1:04d}_{team2:04d}"


def validate_submission(submission_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict:
    """
    Validate submission format against sample.

    Args:
        submission_df: Your submission DataFrame
        sample_df: Sample submission DataFrame

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check columns
    if not all(col in submission_df.columns for col in ['ID', 'Pred']):
        results['valid'] = False
        results['errors'].append("Missing required columns: ID, Pred")
        return results

    # Check number of predictions
    if len(submission_df) != len(sample_df):
        results['valid'] = False
        results['errors'].append(f"Wrong number of predictions: {len(submission_df)} vs {len(sample_df)} expected")

    # Check ID format
    invalid_ids = submission_df[~submission_df['ID'].str.match(r'^\d{4}_\d{4}_\d{4}$')]
    if len(invalid_ids) > 0:
        results['valid'] = False
        results['errors'].append(f"Invalid ID format in {len(invalid_ids)} rows")

    # Check prediction range
    if submission_df['Pred'].min() < 0 or submission_df['Pred'].max() > 1:
        results['valid'] = False
        results['errors'].append("Predictions must be between 0 and 1")

    # Check for missing values
    if submission_df.isnull().any().any():
        results['valid'] = False
        results['errors'].append("Submission contains missing values")

    # Check for duplicate IDs
    if submission_df['ID'].duplicated().any():
        results['valid'] = False
        results['errors'].append("Submission contains duplicate IDs")

    # Warnings
    if (submission_df['Pred'] == 0.5).sum() > len(submission_df) * 0.5:
        results['warnings'].append("More than 50% of predictions are exactly 0.5")

    return results


# ==================== SEED ANALYSIS HELPERS ====================

def parse_seed(seed_str: str) -> Tuple[str, int]:
    """
    Parse seed string into region and numeric seed.

    Args:
        seed_str: Seed string (e.g., 'W01', 'X16a')

    Returns:
        Tuple of (region, seed_number)
    """
    region = seed_str[0]
    seed_num = int(seed_str[1:3])
    return region, seed_num


def get_seed_matchup_label(seed1: int, seed2: int) -> str:
    """
    Create readable matchup label from seeds.

    Args:
        seed1: First team's seed number
        seed2: Second team's seed number

    Returns:
        Matchup label (e.g., '1 vs 16', '8 vs 9')
    """
    low, high = min(seed1, seed2), max(seed1, seed2)
    return f"{low} vs {high}"


# ==================== VISUALIZATION HELPERS ====================

def plot_seed_performance_matrix(tourney_df: pd.DataFrame, seeds_df: pd.DataFrame,
                                 figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create heatmap showing historical win rates for each seed matchup.

    Args:
        tourney_df: Tournament results DataFrame
        seeds_df: Seeds DataFrame
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Merge seeds with results
    df = tourney_df.merge(
        seeds_df[['Season', 'TeamID', 'Seed']],
        left_on=['Season', 'WTeamID'],
        right_on=['Season', 'TeamID']
    ).merge(
        seeds_df[['Season', 'TeamID', 'Seed']],
        left_on=['Season', 'LTeamID'],
        right_on=['Season', 'TeamID'],
        suffixes=('_W', '_L')
    )

    # Extract numeric seeds
    df['Seed_W_Num'] = df['Seed_W'].str[1:3].astype(int)
    df['Seed_L_Num'] = df['Seed_L'].str[1:3].astype(int)

    # Create matrix
    matrix = np.zeros((16, 16))
    counts = np.zeros((16, 16))

    for _, row in df.iterrows():
        sw, sl = row['Seed_W_Num'] - 1, row['Seed_L_Num'] - 1
        matrix[sw, sl] += 1
        counts[sw, sl] += 1
        counts[sl, sw] += 1

    # Calculate win rates
    win_rate = np.divide(matrix, counts, where=counts>0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(win_rate, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=range(1, 17), yticklabels=range(1, 17),
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Opponent Seed')
    ax.set_ylabel('Team Seed')
    ax.set_title('Historical Win Rate by Seed Matchup')

    return fig


def plot_score_distribution(games_df: pd.DataFrame, title: str = "Score Distribution") -> plt.Figure:
    """
    Plot score and margin distributions.

    Args:
        games_df: Game results DataFrame
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Winning score
    axes[0].hist(games_df['WScore'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Winning Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Winning Score Distribution')
    axes[0].axvline(games_df['WScore'].mean(), color='red', linestyle='--',
                    label=f'Mean: {games_df["WScore"].mean():.1f}')
    axes[0].legend()

    # Losing score
    axes[1].hist(games_df['LScore'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Losing Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Losing Score Distribution')
    axes[1].axvline(games_df['LScore'].mean(), color='red', linestyle='--',
                    label=f'Mean: {games_df["LScore"].mean():.1f}')
    axes[1].legend()

    # Margin
    margin = games_df['WScore'] - games_df['LScore']
    axes[2].hist(margin, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[2].set_xlabel('Victory Margin')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Victory Margin Distribution')
    axes[2].axvline(margin.mean(), color='red', linestyle='--',
                    label=f'Mean: {margin.mean():.1f}')
    axes[2].legend()

    plt.suptitle(title)
    plt.tight_layout()

    return fig


# ==================== DATA QUALITY HELPERS ====================

def check_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print data quality report.

    Args:
        df: DataFrame to check
        name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"Data Quality Report: {name}")
    print(f"{'='*60}")

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print(f"Columns are:")

    print(df.columns)
        

    print(f"")

    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())

    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values ✓")
    else:
        missing_pct = 100 * missing / len(df)
        missing_df = pd.DataFrame({'Count': missing[missing > 0],
                                   'Percentage': missing_pct[missing > 0]})
        print(missing_df)

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\n{'='*60}\n")


def get_season_summary(games_df: pd.DataFrame, season: int) -> dict:
    """
    Get summary statistics for a specific season.

    Args:
        games_df: Games DataFrame
        season: Season year

    Returns:
        Dictionary with season statistics
    """
    season_games = games_df[games_df['Season'] == season]

    return {
        'season': season,
        'num_games': len(season_games),
        'num_teams': len(set(season_games['WTeamID'].unique()) | set(season_games['LTeamID'].unique())),
        'avg_winning_score': season_games['WScore'].mean(),
        'avg_losing_score': season_games['LScore'].mean(),
        'avg_margin': (season_games['WScore'] - season_games['LScore']).mean(),
        'num_overtimes': season_games[season_games['NumOT'] > 0]['NumOT'].sum(),
        'overtime_pct': 100 * (season_games['NumOT'] > 0).sum() / len(season_games)
    }


if __name__ == "__main__":
    print("Utility functions loaded successfully")
