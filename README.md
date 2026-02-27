# NCAA March Madness 2026 - Prediction Project

Kaggle Competition: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

## 🎯 Objective

Predict the probability of each possible NCAA tournament matchup using historical game data, team statistics, and advanced metrics.

## 📁 Project Structure

```
NCAA/
├── data/                          # Raw Kaggle data (35 CSV files)
│   ├── MTeams.csv                # Men's teams
│   ├── WTeams.csv                # Women's teams
│   ├── MRegularSeasonCompactResults.csv
│   ├── MNCAATourneyCompactResults.csv
│   └── ...                       # All other data files
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_eda_basic.ipynb        # ✓ Teams, seasons, basic stats
│   ├── 02_eda_tournaments.ipynb   # Tournament & seed analysis (TODO)
│   ├── 03_eda_detailed_stats.ipynb # Box scores & advanced metrics (TODO)
│   ├── 04_eda_rankings.ipynb      # Massey ordinals analysis (TODO)
│   ├── 05_feature_engineering.ipynb # Create & save features (TODO)
│   ├── 06_baseline_model.ipynb    # Simple seed-based model (TODO)
│   ├── 07_modeling.ipynb          # Main ML models (TODO)
│   └── 08_submission.ipynb        # Generate predictions (TODO)
│
├── src/                           # Shared Python modules
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # ✓ Data loading utilities
│   ├── features.py               # ✓ Feature engineering functions
│   └── utils.py                  # ✓ Helper functions
│
├── processed/                     # Intermediate outputs
│   ├── basic_stats.csv           # Summary statistics (from 01_eda_basic)
│   ├── team_features.parquet     # Engineered features (TODO)
│   └── elo_ratings.csv           # Elo ratings over time (TODO)
│
├── models/                        # Saved models
│   └── (model files will go here)
│
└── submissions/                   # Submission files
    └── (submission CSVs will go here)
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
cd /home/sagemaker-user/NCAA
```

### 2. Start with Basic EDA

Open and run `notebooks/01_eda_basic.ipynb` to understand:
- Data coverage (seasons, teams, games)
- Score distributions
- Home court advantage
- Tournament vs regular season patterns

### 3. Use the Shared Modules

In any notebook:

```python
import sys
sys.path.append('/home/sagemaker-user/NCAA')

from src.data_loader import load_teams, load_tourney_results, load_seeds
from src.features import calculate_team_season_stats, calculate_elo_ratings
from src.utils import check_data_quality, create_submission_id

# Load data
teams = load_teams('M')
tourney = load_tourney_results('M', detailed=True)

# Calculate features
team_stats = calculate_team_season_stats(tourney)

# Validate data
check_data_quality(teams, "Teams")
```

## 📚 Module Reference

### `src/data_loader.py`

Load all competition data with consistent interface:

**Basic Data:**
- `load_teams(gender='M')` - Team information
- `load_seasons(gender='M')` - Season dates and regions
- `load_regular_season_results(gender='M', detailed=False)` - Regular season games
- `load_tourney_results(gender='M', detailed=False)` - Tournament games
- `load_seeds(gender='M')` - Tournament seeds

**Enhanced Data:**
- `load_massey_ordinals()` - 40+ ranking systems (men only)
- `load_team_conferences(gender='M')` - Conference affiliations
- `load_game_cities(gender='M')` - Game locations (2010+)

**Helpers:**
- `load_all_games(gender='M', detailed=False)` - Combined regular + tournament
- `validate_data_coverage(gender='M')` - Data coverage summary

### `src/features.py`

Feature engineering with time-awareness (no data leakage):

**Basic Statistics:**
- `calculate_team_season_stats(games_df)` - Win%, avg margin, etc.
- `calculate_seed_features(seeds_df)` - Extract numeric seeds and regions

**Advanced Metrics:**
- `calculate_four_factors(detailed_df)` - Dean Oliver's Four Factors
- `calculate_elo_ratings(games_df, k_factor=32)` - Elo ratings over time

**Matchup Features:**
- `create_matchup_features(season, team1, team2, features_df)` - Features for a specific matchup

**Rankings:**
- `get_latest_massey_rankings(massey_df, season, day_num=133)` - Pre-tournament rankings

### `src/utils.py`

General utilities:

**Submission:**
- `create_submission_id(season, team1, team2)` - Format: `SSSS_XXXX_YYYY`
- `validate_submission(submission_df, sample_df)` - Check submission format

**Analysis:**
- `parse_seed(seed_str)` - Extract region and number from seed
- `get_seed_matchup_label(seed1, seed2)` - Format: "1 vs 16"

**Visualization:**
- `plot_seed_performance_matrix(tourney_df, seeds_df)` - Win rate heatmap
- `plot_score_distribution(games_df)` - Score and margin distributions

**Quality:**
- `check_data_quality(df, name)` - Print data quality report
- `get_season_summary(games_df, season)` - Season statistics

## 🔄 Workflow

### Phase 1: EDA (Notebooks 01-04)
- Understand data structure and patterns
- Identify important features
- Spot anomalies or data quality issues

### Phase 2: Feature Engineering (Notebook 05)
- Use `src/features.py` to calculate metrics
- Save features to `processed/` as Parquet/CSV
- Ensure time-awareness (no data leakage)

### Phase 3: Modeling (Notebooks 06-07)
- Start with simple baseline (seed-based)
- Load pre-computed features from `processed/`
- Train ML models (logistic regression, XGBoost, etc.)
- Save models to `models/`

### Phase 4: Submission (Notebook 08)
- Load trained models
- Generate predictions for all matchups
- Validate submission format
- Save to `submissions/`

## 📊 Submission Format

```csv
ID,Pred
2026_1101_1102,0.5234
2026_1101_1103,0.6891
...
```

Where:
- `ID`: Format `SEASON_TEAMID1_TEAMID2` (lower ID first)
- `Pred`: Win probability for Team1 (0-1)

## 💡 Tips

1. **Always use the shared modules** - Don't duplicate data loading logic
2. **Save intermediate results** - Feature engineering is expensive
3. **Check data coverage** - Not all data goes back to 1985
4. **Mind the timeline** - Use only data available before the game (RankingDayNum < GameDayNum)
5. **Start simple** - A seed-based baseline is surprisingly strong

## 📝 Next Steps

1. ✅ Run `notebooks/01_eda_basic.ipynb`
2. Create `notebooks/02_eda_tournaments.ipynb` for seed analysis
3. Analyze detailed box scores in `notebooks/03_eda_detailed_stats.ipynb`
4. Explore Massey rankings in `notebooks/04_eda_rankings.ipynb`
5. Build baseline model

## 🔗 Resources

- [Competition Page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- [Data Description](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
- [Massey Ratings](https://www.masseyratings.com/cb/ncaa-d1/games)

---

**Good luck! 🏀**
