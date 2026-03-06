# NCAA March Madness 2026 - Tournament Prediction Model

**Kaggle Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

Predicting win probabilities for every possible NCAA tournament matchup (Men's and Women's) using historical game data, engineered features, and an ensemble of gradient-boosted and linear models.

---

## Why This Project

Every March, 68 college basketball teams compete in the NCAA tournament — a single-elimination bracket where upsets are frequent and prediction is genuinely hard. This project tackles the problem as a **probability estimation task**: for any two teams, what is the likelihood that Team A beats Team B?

The challenge isn't just accuracy — it's **calibration**. Saying a team has a 70% chance of winning means they should win roughly 7 out of 10 times. The competition scores submissions using **Brier Score**, which penalizes both overconfidence and underconfidence.

### What Makes This Interesting

- **No leakage allowed** — you can only use data available *before* the game is played
- **Small signal in noisy data** — even the best teams lose to 15-seeds occasionally
- **Probability, not classification** — predicting 0.65 vs 0.85 matters enormously for scoring
- **Two separate pipelines** — men's and women's tournaments have different data availability

---

## Approach

### Core Idea: Cross-Season Prediction

The model is built on a simple but careful principle: **use last season's stats to predict this season's games**.

```
Training Data (no leakage):
  2003 team stats  -->  predict 2004 regular season games
  2004 team stats  -->  predict 2005 regular season games
  ...
  2023 team stats  -->  predict 2024 regular season games
  + Tournament games (same-season stats, since regular season is complete)

Validation:
  2024 team stats  -->  predict 2025 tournament (67 games, known results)

Final Prediction:
  2025 team stats  -->  predict 2026 tournament (unknown results)
```

This gives ~118,000 training matchups while maintaining strict temporal separation.

---

## Feature Engineering

### Raw Data Sources

| Data File | What It Contains | Seasons |
|-----------|-----------------|---------|
| `MRegularSeasonDetailedResults.csv` | Box scores (FGM, FGA, rebounds, turnovers, etc.) | 2003-2026 |
| `MNCAATourneyCompactResults.csv` | Tournament game results | 2003-2025 |
| `MMasseyOrdinals.csv` | 196 ranking systems (Pomeroy, Sagarin, RPI, etc.) | 2003-2025 |
| `MNCAATourneySeeds.csv` | Tournament seeds (1-16 per region) | 2003-2025 |
| `MTeamConferences.csv` | Conference affiliations | 2003-2026 |
| `MConferenceTourneyGames.csv` | Conference tournament results | 2003-2025 |
| `WRegularSeasonDetailedResults.csv` | Women's box scores | 2010-2026 |

### Engineered Features (Per Team, Per Season)

Starting from raw game-level data, I built **18 team-level features** across 5 categories:

**1. Basic Performance**
| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `WinPct` | Wins / Games | Overall team quality |
| `AvgPointsFor` | Total points scored / Games | Offensive output |
| `AvgPointsAgainst` | Total points allowed / Games | Defensive quality |
| `AvgMargin` | AvgPointsFor - AvgPointsAgainst | Dominance level |

**2. Dean Oliver's Four Factors** (from detailed box scores)

These four statistics explain ~90% of winning in basketball:

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `eFG_Off` | (FGM + 0.5 * FGM3) / FGA | Shooting efficiency (adjusts for 3-pointers) |
| `TORate_Off` | TO / (FGA + 0.44 * FTA + TO) | Ball security (lower = better) |
| `ORBRate_Off` | OR / (OR + Opp_DR) | Second-chance scoring opportunities |
| `FTRate_Off` | FTM / FGA | Free throw generation |

**3. Strength Ratings**
| Feature | Source | What It Captures |
|---------|--------|-----------------|
| `Elo_RegSeason` | Calculated from game results | Strength rating that accounts for opponent quality |
| `POM_Rank` | Massey Ordinals (Pomeroy) | Efficiency-based ranking |
| `SAG_Rank` | Massey Ordinals (Sagarin) | Point-spread-based ranking |
| `Composite_Rank` | Average of all 196 ranking systems | Consensus ranking |

**4. Tournament & Conference Context**
| Feature | Source | What It Captures |
|---------|--------|-----------------|
| `SeedNum` | NCAA seeds (1-16, 100 for unseeded) | Selection committee's assessment |
| `IsSeeded` | Derived from SeedNum | Binary: did the team make the tournament? |
| `ConfStrength_Rank` | Conferences ranked by avg team Elo | Quality of competition faced |
| `ConfTourneyChamp` | Conference tournament results | Won conference tournament (momentum signal) |

**5. Late-Season Momentum**
| Feature | Source | What It Captures |
|---------|--------|-----------------|
| `Last15WinPct` | Last 15 regular season games | Recent form vs full-season average |
| `Last15AvgMargin` | Last 15 regular season games | Trending up or down heading into March |

### From Team Features to Matchup Features

For every matchup (Team A vs Team B), I compute the **differential**:

```
AvgMargin_Diff = Team_A.AvgMargin - Team_B.AvgMargin
Elo_Diff       = Team_A.Elo       - Team_B.Elo
SeedNum_Diff   = Team_A.SeedNum   - Team_B.SeedNum
...
```

A positive differential means Team A is stronger on that metric. The model learns how these differentials map to win probability.

### Correlation Analysis & Feature Selection

After computing all 18 differentials, I ran correlation analysis on tournament matchups and found several highly correlated pairs:

| Pair | Correlation | Action |
|------|------------|--------|
| POM_Rank vs Composite_Rank | 0.98 | Dropped POM_Rank |
| WinPct vs Elo | 0.87 | Dropped WinPct |
| AvgPointsFor/Against vs AvgMargin | Redundant (Margin = For - Against) | Dropped both individual components |
| Last15WinPct vs Last15AvgMargin | 0.82 | Dropped Last15WinPct |
| SAG_Rank | NaN for 2025 | Dropped |

**Final feature set: 12 differential features** (men's) / 11 features (women's, no Massey rankings available).

<!-- Add correlation heatmap image here -->
<!-- ![Correlation Heatmap](images/correlation_heatmap.png) -->

---

## Modeling

### Ensemble Strategy

Instead of relying on a single model, I train three different algorithms and blend their predictions. Different algorithms make different errors — averaging them produces more stable, better-calibrated probabilities.

| Model | Type | Why Include It |
|-------|------|---------------|
| **XGBoost** | Gradient-boosted trees (level-wise) | Captures non-linear feature interactions |
| **LightGBM** | Gradient-boosted trees (leaf-wise) | Different tree structure = different errors |
| **Logistic Regression** | Linear model with sigmoid | Naturally well-calibrated for this problem since features are already differentials |

### Hyperparameters (Tuned to Reduce Overfitting)

The initial XGBoost model was overfitting heavily (best iteration at round 39 out of 1200). Key tuning changes:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `max_depth` | 8 | 4 | Shallower trees = less memorization |
| `learning_rate` | 0.075 | 0.02 | Slower learning = better generalization |
| `reg_alpha` | 0 | 0.5 | L1 regularization to zero out weak splits |
| `reg_lambda` | 1 | 3.0 | L2 regularization to shrink leaf weights |
| `min_child_weight` | 3 | 5 | Require more data per leaf |

### Blend Weight Optimization

I grid-searched all weight combinations (step 0.1) on the 2025 tournament validation set to find the optimal blend:

```
Final_Pred = w_XGB * XGBoost_Pred + w_LR * LogReg_Pred + w_LGB * LightGBM_Pred
```

The optimal weights are determined automatically by minimizing Brier Score on held-out 2025 tournament games.

<!-- Add training curves image here -->
<!-- ![Training Curves](images/training_curves.png) -->

---

## Results

### 2025 Tournament Validation (67 games)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **Brier Score** | ~0.16 | Well below 0.25 (coin flip baseline) |
| **Log Loss** | ~0.50 | Model is providing useful probability estimates |
| **Accuracy** | ~77% | Correctly picks the winner in 3 out of 4 games |

### What the Model Gets Right

- **Dominant favorites**: Auburn vs Alabama St, Duke vs Mt St Mary's (>90% confidence, correct)
- **Tournament seeds**: Seed differential is the single strongest predictor
- **Conference strength**: Teams from strong conferences (SEC, Big 12) rated appropriately

### Where the Model Struggles

- **Upsets**: Clemson vs McNeese St (predicted 72% Clemson, McNeese won) — the nature of March Madness
- **Close matchups**: Games between evenly-matched teams (predictions near 0.5) are inherently unpredictable

<!-- Add calibration plot image here -->
<!-- ![Calibration Plot](images/calibration_plot.png) -->

<!-- Add feature importance image here -->
<!-- ![Feature Importance](images/feature_importance.png) -->

---

### Pipeline Flow

```
DATA PROCESSING:
  Raw CSV files --> data_loader.py --> features.py --> team_features.csv

MEN'S PIPELINE:
  04a (build features) --> 04b (build matchups) --> 05b (train ensemble)

WOMEN'S PIPELINE:
  04a_W (build features) --> 04b_W (build matchups) --> 05b_W (train ensemble)

FINAL:
  06 (combine men's + women's predictions) --> submission_final.csv
```

---

## Data & Methodology Details

### Training / Validation / Prediction Split

| Split | Seasons | Games | Purpose |
|-------|---------|-------|---------|
| Training | 2003-2024 | ~108,000 matchups | Learn feature-to-outcome relationships |
| Validation | 2025 | 67 tournament games | Tune hyperparameters, select blend weights |
| Prediction | 2026 | 132,133 possible matchups | Generate Kaggle submission |

### Why Two Separate Pipelines (Men's vs Women's)

| | Men's | Women's |
|--|-------|---------|
| Detailed data starts | 2003 | 2010 |
| Massey Ordinals (196 ranking systems) | Available | Not available |
| Final features after correlation drops | 12 | 11 |
| Training matchups | ~108,000 | ~75,000 |

Women's basketball doesn't have Massey Ordinal rankings published, so the women's model relies more heavily on Elo ratings and box score statistics. Despite fewer features, the approach is identical: engineer team-level stats, compute matchup differentials, train an ensemble, blend predictions.

### Key Design Decisions

1. **Cross-season approach**: Using Season S-1 stats to predict Season S games avoids any data leakage. Tournament predictions use same-season stats (regular season is complete before tournament starts).

2. **SeedNum = 100 for unseeded teams**: Only 68 of ~360 teams per season receive tournament seeds (1-16). Unseeded teams are assigned SeedNum=100 to create a clear separation from the worst seeded team (16). An additional `IsSeeded` binary feature captures the tournament-qualifier signal.

3. **Dropping SAG_Rank entirely**: Sagarin rankings were not published for the 2025 season, making this feature unusable for validation and prediction. Since Composite_Rank already averages all ranking systems including SAG, dropping it had minimal impact.

4. **Equal-weight starting point for ensemble**: Rather than hand-tuning weights, I grid-searched all possible weight combinations on the validation set. This prevents overfitting to any single model's strengths on the training data.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| pandas / NumPy | Data processing |
| XGBoost 3.2 | Gradient-boosted trees (level-wise) |
| LightGBM 4.6 | Gradient-boosted trees (leaf-wise) |
| scikit-learn | Logistic Regression, StandardScaler, metrics |
| matplotlib / seaborn | Visualization |
| Jupyter Notebooks | Analysis pipeline |
| AWS SageMaker | Compute environment |

---

## How to Reproduce

```bash
# 1. Clone and navigate
cd NCAA_March_Maddness-2026

# 2. Run men's pipeline
jupyter nbconvert --execute notebooks/04a_feature_building.ipynb
jupyter nbconvert --execute notebooks/04b_matchup_building.ipynb
jupyter nbconvert --execute notebooks/05b_ensemble_modeling.ipynb

# 3. Run women's pipeline
jupyter nbconvert --execute notebooks/04a_W_feature_building.ipynb
jupyter nbconvert --execute notebooks/04b_W_matchup_building.ipynb
jupyter nbconvert --execute notebooks/05b_W_ensemble_modeling.ipynb

# 4. Generate final submission
jupyter nbconvert --execute notebooks/06_final_submission.ipynb

# Output: submissions/submission_final.csv (132,133 rows)
```

---

## Visualizations

<!-- Replace these placeholders with your actual visualization images -->

### Feature Distributions (Winners vs Losers)
<!-- ![Feature Distributions](images/feature_distributions.png) -->
*Tournament matchups split by outcome. Greater separation between green (winner) and red (loser) distributions indicates stronger predictive power.*

### Correlation Heatmap
<!-- ![Correlation Heatmap](images/correlation_heatmap.png) -->
*Feature correlations in tournament matchups. Highly correlated pairs (|r| > 0.8) were identified and redundant features dropped to reduce multicollinearity.*

### Training Curves
<!-- ![Training Curves](images/training_curves.png) -->
*XGBoost training vs validation loss. Early stopping prevents overfitting — the gap between curves shows where the model starts memorizing noise.*

### Calibration Plot
<!-- ![Calibration Plot](images/calibration_plot.png) -->
*Model calibration check: predicted probabilities vs actual win rates. Points close to the diagonal mean the model's confidence matches reality.*

### Feature Importance
<!-- ![Feature Importance](images/feature_importance.png) -->
*Feature importance comparison across all three models. Each algorithm weights features differently, which is why blending them produces more robust predictions.*

---

## Future Improvements

- **Hyperparameter tuning with Optuna** — automated Bayesian search instead of manual tuning
- **Isotonic calibration** — post-hoc calibration layer on top of the ensemble
- **Additional features** — coach experience, travel distance, rest days between games
- **Stage 2 re-training** — incorporate 2026 regular season data once available for final tournament predictions

---

**Competition Link:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
