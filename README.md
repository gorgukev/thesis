# Predicting Over 2.5 Goals in Football Matches Using Machine Learning

A bachelor's thesis project investigating whether pre-match statistics can predict high-scoring football matches (over 2.5 goals) in the English Premier League.

## 📊 Project Overview

This project explores the feasibility of predicting entertaining football matches (defined as matches with over 2.5 goals) using only information available before kick-off. The goal is to create a decision support system for football fans who want to prioritize which matches to watch based on the likelihood of goals.

**Research Question**: *How well can we predict the probability that a match will end with over 2.5 goals using only pre-match information?*

## 🎯 Key Findings

- **ROC-AUC**: ~0.50 (barely better than random guessing)
- **Accuracy**: ~52% (not significantly better than baseline)
- **Brier Score**: 0.25 (uncalibrated probabilities)
- **Precision@5**: 0.60 (60% of top-5 predictions were correct)

**Conclusion**: Simple pre-match statistics (rolling averages of goals, points, early goals) are **insufficient** for reliable predictions. More sophisticated features (xG, tactical data, player injuries) would be needed.

## 📁 Project Structure

```
├── Collect_data.ipynb          # Main Jupyter notebook (all 5 parts)
├── fixture_meta_data.csv       # Basic match metadata (3800 matches)
├── goals_prem_2015_2024.csv    # Detailed goal timing data
├── complete_fixture_data.csv   # Merged dataset
├── README.md                   # This file
└── Examensarbete.docx          # Thesis document (Swedish)
```

## 🔧 Technologies Used

- **Python**: 3.14.2
- **pandas**: 2.3.3 - Data manipulation
- **NumPy**: 2.4.0 - Numerical operations
- **scikit-learn**: 1.8.0 - Machine learning models
- **Football-API**: Data source (https://www.api-football.com/)

## 📚 Dataset

- **League**: English Premier League
- **Seasons**: 2015-2025 (10 seasons)
- **Total matches**: 3,800
- **Training set**: Seasons 2015-2022 (3,018 matches)
- **Test set**: Seasons 2023-2024 (758 matches)
- **Class balance**: 54% over 2.5 goals, 46% under 2.5 goals

### Data Sources
All data collected from Football-API:
1. Match metadata (teams, scores, dates)
2. Detailed goal events (minute-by-minute)

## 🚀 Installation & Setup

### Prerequisites
```bash
# Python 3.10+
pip install pandas numpy scikit-learn jupyter
```

### API Setup
1. Register at [Football-API](https://www.api-football.com/)
2. Get your API key (free tier: 100 requests/day)
3. Add your key to the notebook:
```python
API_KEY = "your_api_key_here"
```

### Running the Project
```bash
# Open Jupyter notebook
jupyter notebook Collect_data.ipynb

# Run all cells sequentially
# Note: Part 2 (data collection) requires API access and takes time, the API key in the project is expired.
```

## 📖 Notebook Structure

The project is divided into 5 parts:

### **Part 1: Data Collection**
- Fetches basic match information from Football-API
- Creates `fixture_meta_data.csv`
- Adds target variable: `fulltime_over_2_5` (1/0)

### **Part 2: Detailed Goal Statistics**
- Fetches minute-by-minute goal data for each match
- Implements checkpoint system for resume capability
- Handles API rate limits and retries
- Creates `goals_prem_2015_2024.csv`

### **Part 3: Data Merging**
- Combines metadata with goal timing data
- Creates lists of goal minutes per team per match
- Preserves 0-0 matches (empty lists)
- Creates `complete_fixture_data.csv`

### **Part 4: Feature Engineering**
- Calculates rolling averages (window=3) for each team
- **Critical**: Uses `.shift(1)` to avoid data leakage
- Features created (18 total):
  - Points (average and sum last 3 matches)
  - Goals scored/conceded (halftime and fulltime)
  - Early goals (0-15 minutes, first half)
- Final dataset: 3,776 matches (24 dropped due to missing history)

### **Part 5: Model Training & Evaluation**
- **Logistic Regression**: Baseline model (linear, interpretable)
- **Random Forest**: Non-linear alternative (500 trees, max_depth=6)
- **Temporal split**: Last 2 seasons as test set
- **Evaluation metrics**: ROC-AUC, Accuracy, Brier Score, Precision@5

## 🎓 Features Explained

All features are **pre-match only** to avoid data leakage:

| Feature | Description | Example |
|---------|-------------|---------|
| `home_avg_ft_gf_last3` | Home team's avg goals scored (last 3 matches) | 1.67 |
| `away_avg_ft_ga_last3` | Away team's avg goals conceded (last 3 matches) | 2.33 |
| `home_pts_sum_last3` | Home team's total points (last 3 matches) | 7 |
| `home_avg_scored_0_15_last3` | Home team's avg goals in min 0-15 | 0.33 |
| `away_avg_scored_1H_last3` | Away team's avg first-half goals | 1.0 |

**Why rolling averages?**
- Captures recent form without using future information
- `.shift(1)` ensures match N only uses data from matches 1 to N-1
- Window of 3 balances recency vs. sample size

## ⚠️ Critical Design Decisions

### 1. Temporal Split (Not Random Split)
```python
test_seasons = [2023, 2024]  # Last 2 seasons
```
**Why?** Simulates real-world scenario: predict future matches using historical data.

### 2. Data Leakage Prevention
```python
# ❌ WRONG: Uses current match result
df["home_goals"] = df["ft_home_goals"]

# ✅ CORRECT: Uses average of previous matches
df["home_avg_goals"] = df.groupby("team_id")["ft_home_goals"]\
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
```

### 3. Class Imbalance Handling
```python
LogisticRegression(class_weight='balanced')
RandomForestClassifier(class_weight='balanced')
```
Compensates for 54/46 class split.

### 4. Checkpoint System
```python
checkpoint_path = save_path + ".done"
```
Allows resuming data collection after interruptions (crucial with API rate limits).

## 📊 Results Interpretation

### Why Did the Models Fail?

**Missing Critical Information**:
1. **xG (Expected Goals)**: Quality of chances, not just quantity
2. **Tactical matchups**: Offensive vs. defensive playing styles
3. **Player availability**: Injuries, suspensions, rotation
4. **Motivation**: Title races, relegation battles, derbies
5. **Match context**: Weather, referee tendencies, recent head-to-heads

**Fundamental Limitation**:
Football has high **inherent randomness**. Even top bookmakers struggle to exceed 55-60% accuracy for over/under predictions.

### Comparison with Previous Research

According to Rathke (2017), most prediction models achieve ~60% accuracy. Our models achieved ~52%, indicating:
- The chosen features are not sufficiently predictive
- More advanced methods (xG-based models, neural networks) needed

## 🔮 Future Work

### Short-term Improvements
1. **Probability calibration**: Improve confidence estimates
2. **Feature importance analysis**: Identify which features matter most
3. **Hyperparameter tuning**: Grid search for optimal parameters
4. **Additional leagues**: Expand dataset to Bundesliga, La Liga, Serie A

### Long-term Research Directions
1. **xG integration**: Incorporate expected goals data
2. **Player-level features**: Team strength based on starting XI
3. **Tactical indicators**: Pressing intensity, possession style
4. **Deep learning**: LSTM/GRU for temporal patterns
5. **Ensemble methods**: Combine multiple models

## 📄 License

This project is submitted as a bachelor's thesis at Arcada University of Applied Sciences. 
For academic use, please cite appropriately according to the MIT license.

## 👤 Author

**[Kevin Görgü]**
- Bachelor's Thesis in [Information technology]
- Arcada University of Applied Sciences
- Year: 2026

## 🙏 Acknowledgments

- **Football-API**: For providing comprehensive football data
- **GoalStatistics**: For insights on early goal correlations
- **Arcada University**: For academic support and guidance

## 📖 References

Key references used in this project:
For complete references, see the thesis in theseum: **Link incoming**.


---

**Note**: This project demonstrates that simple statistical approaches have **limited predictive power** for football match outcomes. While the results are not suitable for a production system, the methodology provides a solid foundation for more advanced research incorporating richer features and more sophisticated modeling techniques.
