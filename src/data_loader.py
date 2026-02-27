"""
Data loading utilities for NCAA March Madness competition.
Provides consistent interface to load all data files with optional caching.
"""

import pandas as pd
from pathlib import Path
from typing import Literal

# Project paths
DATA_DIR = Path("/home/sagemaker-user/NCAA/data")
PROCESSED_DIR = Path("/home/sagemaker-user/NCAA/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

GenderType = Literal['M', 'W']


# ==================== SECTION 1: BASIC DATA ====================

def load_teams(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load teams data.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with columns: TeamID, TeamName, FirstD1Season (men only), LastD1Season (men only)
    """
    return pd.read_csv(DATA_DIR / f"{gender}Teams.csv")


def load_seasons(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load seasons data with DayZero reference dates and regions.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with columns: Season, DayZero, RegionW, RegionX, RegionY, RegionZ
    """
    return pd.read_csv(DATA_DIR / f"{gender}Seasons.csv")


def load_regular_season_results(gender: GenderType = 'M', detailed: bool = False) -> pd.DataFrame:
    """
    Load regular season game results.

    Args:
        gender: 'M' for men's, 'W' for women's
        detailed: If True, loads detailed box score stats (2003+ for men, 2010+ for women)

    Returns:
        DataFrame with game results. Detailed version includes FGM, FGA, rebounds, etc.
    """
    suffix = "Detailed" if detailed else "Compact"
    return pd.read_csv(DATA_DIR / f"{gender}RegularSeason{suffix}Results.csv")


def load_tourney_results(gender: GenderType = 'M', detailed: bool = False) -> pd.DataFrame:
    """
    Load NCAA tournament game results.

    Args:
        gender: 'M' for men's, 'W' for women's
        detailed: If True, loads detailed box score stats

    Returns:
        DataFrame with tournament game results
    """
    suffix = "Detailed" if detailed else "Compact"
    return pd.read_csv(DATA_DIR / f"{gender}NCAATourney{suffix}Results.csv")


def load_seeds(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load tournament seeds.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with columns: Season, Seed (e.g., 'W01'), TeamID
    """
    return pd.read_csv(DATA_DIR / f"{gender}NCAATourneySeeds.csv")


# ==================== SECTION 2: ENHANCED DATA ====================

def load_massey_ordinals() -> pd.DataFrame:
    """
    Load Massey ordinal rankings (men's only).
    Contains 40+ ranking systems (Pomeroy, Sagarin, RPI, etc.)

    Returns:
        DataFrame with columns: Season, RankingDayNum, SystemName, TeamID, OrdinalRank
    """
    return pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")


def load_team_conferences(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load team conference affiliations by season.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with columns: Season, TeamID, ConfAbbrev
    """
    return pd.read_csv(DATA_DIR / f"{gender}TeamConferences.csv")


def load_conferences() -> pd.DataFrame:
    """
    Load conference names and abbreviations.

    Returns:
        DataFrame with columns: ConfAbbrev, Description
    """
    return pd.read_csv(DATA_DIR / "Conferences.csv")


# ==================== SECTION 3: GEOGRAPHY ====================

def load_cities() -> pd.DataFrame:
    """
    Load city reference data.

    Returns:
        DataFrame with columns: CityID, City, State
    """
    return pd.read_csv(DATA_DIR / "Cities.csv")


def load_game_cities(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load game location data (2010+ seasons).

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with columns: Season, DayNum, WTeamID, LTeamID, CRType, CityID
    """
    return pd.read_csv(DATA_DIR / f"{gender}GameCities.csv")


# ==================== SECTION 4: SUPPLEMENTAL DATA ====================

def load_conference_tourney_games(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load conference tournament game identifiers.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with columns: Season, DayNum, WTeamID, LTeamID, ConfAbbrev
    """
    return pd.read_csv(DATA_DIR / f"{gender}ConferenceTourneyGames.csv")


def load_secondary_tourney_results(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load secondary tournament results (NIT, CBI, etc.).

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame with game results from secondary tournaments
    """
    return pd.read_csv(DATA_DIR / f"{gender}SecondaryTourneyCompactResults.csv")


def load_tourney_slots(gender: GenderType = 'M') -> pd.DataFrame:
    """
    Load tournament bracket structure.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        DataFrame defining bracket slots and matchups
    """
    return pd.read_csv(DATA_DIR / f"{gender}NCAATourneySlots.csv")


# ==================== SECTION 5: SUBMISSION HELPERS ====================

def load_sample_submission(stage: int = 1) -> pd.DataFrame:
    """
    Load sample submission file.

    Args:
        stage: 1 for Stage 1 (historical), 2 for Stage 2 (current season)

    Returns:
        DataFrame with columns: ID (format: SSSS_XXXX_YYYY), Pred
    """
    return pd.read_csv(DATA_DIR / f"SampleSubmissionStage{stage}.csv")


# ==================== SECTION 6: COMBINED LOADERS ====================

def load_all_games(gender: GenderType = 'M', detailed: bool = False) -> pd.DataFrame:
    """
    Load all games (regular season + NCAA tournament) combined.

    Args:
        gender: 'M' for men's, 'W' for women's
        detailed: If True, loads detailed box scores

    Returns:
        Combined DataFrame with a 'GameType' column indicating 'Regular' or 'NCAA'
    """
    regular = load_regular_season_results(gender, detailed)
    regular['GameType'] = 'Regular'

    tourney = load_tourney_results(gender, detailed)
    tourney['GameType'] = 'NCAA'

    return pd.concat([regular, tourney], ignore_index=True)


def load_team_lookup(gender: GenderType = 'M') -> dict:
    """
    Create a dictionary mapping TeamID -> TeamName for easy lookups.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        Dictionary {TeamID: TeamName}
    """
    teams = load_teams(gender)
    return dict(zip(teams['TeamID'], teams['TeamName']))


# ==================== SECTION 7: DATA VALIDATION ====================

def validate_data_coverage(gender: GenderType = 'M') -> dict:
    """
    Validate data coverage and return summary statistics.

    Args:
        gender: 'M' for men's, 'W' for women's

    Returns:
        Dictionary with data coverage information
    """
    teams = load_teams(gender)
    regular = load_regular_season_results(gender)
    tourney = load_tourney_results(gender)
    seeds = load_seeds(gender)

    return {
        'num_teams': len(teams),
        'regular_season_years': sorted(regular['Season'].unique()),
        'tournament_years': sorted(tourney['Season'].unique()),
        'num_regular_games': len(regular),
        'num_tourney_games': len(tourney),
        'seeds_coverage': sorted(seeds['Season'].unique()),
    }


if __name__ == "__main__":
    # Quick test
    print("Testing data loaders...")
    print("\nMen's Teams:", len(load_teams('M')))
    print("Women's Teams:", len(load_teams('W')))
    print("\nMen's Data Coverage:")
    for k, v in validate_data_coverage('M').items():
        print(f"  {k}: {v}")
