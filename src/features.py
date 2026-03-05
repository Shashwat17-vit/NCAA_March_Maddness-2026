"""
Feature engineering utilities for NCAA March Madness prediction.
All feature functions should be time-aware (no data leakage).
"""

import pandas as pd
import numpy as np
from typing import Literal

GenderType = Literal['M', 'W']


# ==================== BASIC TEAM STATISTICS ====================

def calculate_team_season_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic season statistics for each team.

    Args:
        games_df: DataFrame with game results (must have: Season, WTeamID, LTeamID, WScore, LScore)

    Returns:
        DataFrame with columns: Season, TeamID, Wins, Losses, WinPct, AvgMargin, etc.
    """
    stats_list = []

    # Process wins
    wins = games_df.groupby(['Season', 'WTeamID']).agg(
        Wins=('WTeamID', 'count'),
        PointsFor=('WScore', 'sum'),
        PointsAgainst=('LScore', 'sum')
    ).reset_index()
    wins.columns = ['Season', 'TeamID', 'Wins', 'PointsFor', 'PointsAgainst']

    # Process losses
    losses = games_df.groupby(['Season', 'LTeamID']).agg(
        Losses=('LTeamID', 'count'),
        PointsFor=('LScore', 'sum'),
        PointsAgainst=('WScore', 'sum')
    ).reset_index()
    losses.columns = ['Season', 'TeamID', 'Losses', 'PointsForL', 'PointsAgainstL']

    # Merge
    stats = wins.merge(losses, on=['Season', 'TeamID'], how='outer').fillna(0)

    # Calculate aggregates
    stats['Wins'] = stats['Wins'].astype(int)
    stats['Losses'] = stats['Losses'].astype(int)
    stats['Games'] = stats['Wins'] + stats['Losses']
    stats['WinPct'] = stats['Wins'] / stats['Games']
    stats['TotalPointsFor'] = stats['PointsFor'] + stats['PointsForL']
    stats['TotalPointsAgainst'] = stats['PointsAgainst'] + stats['PointsAgainstL']
    stats['AvgPointsFor'] = stats['TotalPointsFor'] / stats['Games']
    stats['AvgPointsAgainst'] = stats['TotalPointsAgainst'] / stats['Games']
    stats['AvgMargin'] = stats['AvgPointsFor'] - stats['AvgPointsAgainst']

    return stats[['Season', 'TeamID', 'Games', 'Wins', 'Losses', 'WinPct',
                  'AvgPointsFor', 'AvgPointsAgainst', 'AvgMargin']]


def calculate_seed_features(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric seed values and region information.

    Args:
        seeds_df: Tournament seeds DataFrame

    Returns:
        DataFrame with numeric seed values and regions
    """
    seeds = seeds_df.copy()

    # Extract region (W, X, Y, Z)
    seeds['Region'] = seeds['Seed'].str[0]

    # Extract numeric seed (01-16)
    seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)

    # Play-in indicator
    seeds['IsPlayIn'] = seeds['Seed'].str.len() > 3

    return seeds[['Season', 'TeamID', 'Seed', 'Region', 'SeedNum', 'IsPlayIn']]


# ==================== ADVANCED METRICS ====================

def calculate_four_factors(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Dean Oliver's Four Factors for each team-game.
    Requires detailed box score data.

    Four Factors:
    1. Effective FG% = (FGM + 0.5 * FGM3) / FGA
    2. Turnover Rate = TO / (FGA + 0.44 * FTA + TO)
    3. Offensive Rebound Rate = OR / (OR + Opp_DR)
    4. Free Throw Rate = FTM / FGA

    Args:
        detailed_df: DataFrame with detailed box scores

    Returns:
        DataFrame with four factors for winning and losing teams
    """
    df = detailed_df.copy()

    # Winning team factors
    df['W_eFG'] = (df['WFGM'] + 0.5 * df['WFGM3']) / df['WFGA']
    df['W_TORate'] = df['WTO'] / (df['WFGA'] + 0.44 * df['WFTA'] + df['WTO'])
    df['W_ORBRate'] = df['WOR'] / (df['WOR'] + df['LDR'])
    df['W_FTRate'] = df['WFTM'] / df['WFGA']

    # Losing team factors
    df['L_eFG'] = (df['LFGM'] + 0.5 * df['LFGM3']) / df['LFGA']
    df['L_TORate'] = df['LTO'] / (df['LFGA'] + 0.44 * df['LFTA'] + df['LTO'])
    df['L_ORBRate'] = df['LOR'] / (df['LOR'] + df['WDR'])
    df['L_FTRate'] = df['LFTM'] / df['LFGA']

    return df


def aggregate_four_factors(four_factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate four factors to team-season level.

    Args:
        four_factors_df: Output from calculate_four_factors()

    Returns:
        DataFrame with season-averaged four factors per team
    """
    # Process winning games
    wins = four_factors_df.groupby(['Season', 'WTeamID']).agg({
        'W_eFG': 'mean',
        'W_TORate': 'mean',
        'W_ORBRate': 'mean',
        'W_FTRate': 'mean'
    }).reset_index()
    wins.columns = ['Season', 'TeamID', 'eFG', 'TORate', 'ORBRate', 'FTRate']

    # Process losing games (defensive perspective)
    losses = four_factors_df.groupby(['Season', 'LTeamID']).agg({
        'L_eFG': 'mean',
        'L_TORate': 'mean',
        'L_ORBRate': 'mean',
        'L_FTRate': 'mean'
    }).reset_index()
    losses.columns = ['Season', 'TeamID', 'eFG_L', 'TORate_L', 'ORBRate_L', 'FTRate_L']

    # Combine
    factors = wins.merge(losses, on=['Season', 'TeamID'], how='outer')

    # Average offensive and defensive (when team both won and lost games)
    factors['eFG_Off'] = factors[['eFG', 'eFG_L']].mean(axis=1, skipna=True)
    factors['TORate_Off'] = factors[['TORate', 'TORate_L']].mean(axis=1, skipna=True)
    factors['ORBRate_Off'] = factors[['ORBRate', 'ORBRate_L']].mean(axis=1, skipna=True)
    factors['FTRate_Off'] = factors[['FTRate', 'FTRate_L']].mean(axis=1, skipna=True)

    return factors[['Season', 'TeamID', 'eFG_Off', 'TORate_Off', 'ORBRate_Off', 'FTRate_Off']]


# ==================== ELO RATING SYSTEM ====================

def calculate_elo_ratings(games_df: pd.DataFrame, k_factor: int = 32,
                         initial_elo: int = 1500, scale: int = 200) -> pd.DataFrame:
    """
    Calculate Elo ratings for all teams over time.

    Args:
        games_df: DataFrame with game results (must be sorted by Season, DayNum)
        k_factor: Elo K-factor (higher = more volatile)
        initial_elo: Starting Elo rating
        scale: Scaling factor (200 = a 200-point gap means 10x more likely to win)

    Returns:
        DataFrame with columns: Season, DayNum, TeamID, Elo
    """
    games = games_df.sort_values(['Season', 'DayNum']).copy()

    # Initialize Elo dictionary
    elo_dict = {}
    elo_history = []

    for _, game in games.iterrows():
        season = game['Season']
        day = game['DayNum']
        team_w = game['WTeamID']
        team_l = game['LTeamID']

        # Get current Elo (reset each season)
        key_w = (season, team_w)
        key_l = (season, team_l)

        elo_w = elo_dict.get(key_w, initial_elo)
        elo_l = elo_dict.get(key_l, initial_elo)

        # Calculate expected win probability
        expected_w = 1 / (1 + 10 ** ((elo_l - elo_w) / scale))

        # Update Elo
        new_elo_w = elo_w + k_factor * (1 - expected_w)
        new_elo_l = elo_l + k_factor * (0 - (1 - expected_w))

        # Store
        elo_dict[key_w] = new_elo_w
        elo_dict[key_l] = new_elo_l

        # Record history (after game)
        elo_history.append({'Season': season, 'DayNum': day, 'TeamID': team_w, 'Elo': new_elo_w})
        elo_history.append({'Season': season, 'DayNum': day, 'TeamID': team_l, 'Elo': new_elo_l})

    return pd.DataFrame(elo_history)


# ==================== CONFERENCE STRENGTH ====================

def calculate_conference_strength(conferences_df: pd.DataFrame,
                                  elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank conferences by average end-of-season Elo of their teams.

    Args:
        conferences_df: DataFrame with Season, TeamID, ConfAbbrev
        elo_df: DataFrame with Season, TeamID, Elo_RegSeason

    Returns:
        DataFrame with Season, TeamID, ConfAbbrev, ConfStrength_Rank
    """
    # Merge conference assignments with Elo ratings
    merged = conferences_df.merge(elo_df, on=['Season', 'TeamID'], how='left')

    # Average Elo per conference per season
    conf_avg = merged.groupby(['Season', 'ConfAbbrev']).agg(
        ConfAvgElo=('Elo_RegSeason', 'mean')
    ).reset_index()

    # Rank conferences per season (1 = highest avg Elo = strongest)
    conf_avg['ConfStrength_Rank'] = conf_avg.groupby('Season')['ConfAvgElo'].rank(
        ascending=False, method='min'
    ).astype(int)

    # Map rank back to each team
    result = conferences_df.merge(
        conf_avg[['Season', 'ConfAbbrev', 'ConfStrength_Rank']],
        on=['Season', 'ConfAbbrev'], how='left'
    )

    return result[['Season', 'TeamID', 'ConfAbbrev', 'ConfStrength_Rank']]


# ==================== LAST N GAMES PERFORMANCE ====================

def calculate_last_n_games(games_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Calculate WinPct and AvgMargin over the last N regular season games.
    Captures late-season momentum/form.

    Args:
        games_df: DataFrame with Season, DayNum, WTeamID, WScore, LTeamID, LScore
        n: Number of recent games to use (default 15)

    Returns:
        DataFrame with Season, TeamID, Last{n}WinPct, Last{n}AvgMargin
    """
    # Build one row per team per game with result
    wins = games_df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore']].copy()
    wins.columns = ['Season', 'DayNum', 'TeamID', 'PointsFor', 'PointsAgainst']
    wins['Win'] = 1

    losses = games_df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore']].copy()
    losses.columns = ['Season', 'DayNum', 'TeamID', 'PointsFor', 'PointsAgainst']
    losses['Win'] = 0

    all_games = pd.concat([wins, losses], ignore_index=True)
    all_games['Margin'] = all_games['PointsFor'] - all_games['PointsAgainst']
    all_games = all_games.sort_values(['Season', 'TeamID', 'DayNum'])

    # Take last N games per team per season
    last_n = all_games.groupby(['Season', 'TeamID']).tail(n)

    # Aggregate
    result = last_n.groupby(['Season', 'TeamID']).agg(
        GamesUsed=('Win', 'count'),
        Wins=('Win', 'sum'),
        AvgMargin=('Margin', 'mean')
    ).reset_index()

    result[f'Last{n}WinPct'] = result['Wins'] / result['GamesUsed']
    result[f'Last{n}AvgMargin'] = result['AvgMargin']

    return result[['Season', 'TeamID', f'Last{n}WinPct', f'Last{n}AvgMargin']]


# ==================== CONFERENCE TOURNAMENT CHAMPION ====================

def calculate_conf_tourney_champ(conf_tourney_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify conference tournament champions.
    The winner of the last game (max DayNum) per conference per season is the champion.

    Args:
        conf_tourney_df: DataFrame with Season, ConfAbbrev, DayNum, WTeamID, LTeamID

    Returns:
        DataFrame with Season, TeamID, ConfTourneyChamp (1=champion, 0=not)
    """
    # Find championship game: last game per conference per season
    champ_games = conf_tourney_df.sort_values('DayNum').groupby(
        ['Season', 'ConfAbbrev']
    ).tail(1)

    # Champions are the winners of those games
    champs = champ_games[['Season', 'WTeamID']].copy()
    champs.columns = ['Season', 'TeamID']
    champs['ConfTourneyChamp'] = 1

    return champs[['Season', 'TeamID', 'ConfTourneyChamp']]


# ==================== MATCHUP FEATURE CREATION ====================

def create_matchup_features(season: int, team1: int, team2: int,
                           team_features: pd.DataFrame,
                           seed_features: pd.DataFrame = None) -> dict:
    """
    Create features for a specific team matchup.

    Args:
        season: Season year
        team1: First team ID
        team2: Second team ID
        team_features: Pre-computed team features
        seed_features: Tournament seed features (optional)

    Returns:
        Dictionary of features for the matchup
    """
    features = {}

    # Get team stats
    t1_stats = team_features[(team_features['Season'] == season) &
                            (team_features['TeamID'] == team1)]
    t2_stats = team_features[(team_features['Season'] == season) &
                            (team_features['TeamID'] == team2)]

    if len(t1_stats) == 0 or len(t2_stats) == 0:
        return None

    t1_stats = t1_stats.iloc[0]
    t2_stats = t2_stats.iloc[0]

    # Differential features
    features['WinPct_Diff'] = t1_stats.get('WinPct', 0) - t2_stats.get('WinPct', 0)
    features['AvgMargin_Diff'] = t1_stats.get('AvgMargin', 0) - t2_stats.get('AvgMargin', 0)
    features['AvgPointsFor_Diff'] = t1_stats.get('AvgPointsFor', 0) - t2_stats.get('AvgPointsFor', 0)

    # Seed features (if available)
    if seed_features is not None:
        s1 = seed_features[(seed_features['Season'] == season) &
                          (seed_features['TeamID'] == team1)]
        s2 = seed_features[(seed_features['Season'] == season) &
                          (seed_features['TeamID'] == team2)]

        if len(s1) > 0 and len(s2) > 0:
            features['Seed1'] = s1.iloc[0]['SeedNum']
            features['Seed2'] = s2.iloc[0]['SeedNum']
            features['Seed_Diff'] = s2.iloc[0]['SeedNum'] - s1.iloc[0]['SeedNum']

    return features


# ==================== MASSEY RANKINGS PROCESSING ====================

def get_latest_massey_rankings(massey_df: pd.DataFrame, season: int,
                               day_num: int = 133) -> pd.DataFrame:
    """
    Get the latest rankings before a specific day (default: pre-tournament).

    Args:
        massey_df: Massey ordinals DataFrame
        season: Season year
        day_num: Get rankings valid for this day (default 133 = pre-tourney)

    Returns:
        DataFrame with latest rankings for each system and team
    """
    # Filter to season and valid days
    valid = massey_df[(massey_df['Season'] == season) &
                     (massey_df['RankingDayNum'] <= day_num)]

    # Get most recent ranking for each system-team combo
    latest = valid.sort_values('RankingDayNum').groupby(['SystemName', 'TeamID']).tail(1)

    return latest[['SystemName', 'TeamID', 'OrdinalRank', 'RankingDayNum']]


if __name__ == "__main__":
    print("Feature engineering utilities loaded successfully")
