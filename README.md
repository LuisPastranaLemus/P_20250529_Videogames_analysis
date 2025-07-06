# 🧭 Project Analysis   

ICE Video Games — Identifying Success Patterns in the Gaming Industry

This project analyzes video game sales data from the global digital retailer **ICE**, aiming to **identify patterns that determine a game's success**. By examining sales, reviews, genres, platforms, and ESRB ratings, we generate actionable insights that can support **marketing strategies** and help **predict promising game releases**.

---

## 🔍 Project Overview (P-20250529_Videogames_analysis)

Project Overview

Uncover what defines a successful video game using historical data — enabling the detection of high-potential projects and the strategic planning of advertising campaigns.

Key questions:

- Identify how many games were released in different years, focus on periods of time.
- Observe how sales vary from one platform to another. 
- Choose the platforms with the highest total sales and construct a distribution based on each year's data. 
- Look for platforms that used to be popular but are now losing sales. 
- How long does it generally take for new platforms to appear and old ones to disappear?
- Determine which period you should collect data for.
- Choose several potentially profitable platforms.
- Show the average sales across various platforms? 
- See how user and professional reviews affect sales for a popular platform (your choice). 
- Create a scatter plot and calculate the correlation between reviews and sales.
- Compare sales of the same games on other platforms.
- Take a look at the overall distribution of games by genre. 
- What can be said about the most profitable genres? 
- Can you generalize about genres with high and low sales?

- Test the following hypotheses:
    - The average user ratings for the Xbox One and PC platforms are the same.
    - The average user ratings for the Action and Sports genres are different.

__Note__: The dataset contains a "rating" column that stores the ESRB rating for each game. The Entertainment Software Rating Board evaluates a game's content and assigns an age rating of either Teen or Mature.

---

## 🧮 Data Dictionary

This project has N different tables.

- `games.csv` (describe content)
- `Name`: video game's name
- `Platform`: video games's platform
- `Year of Release`: video game's realease date
- `Genre`: video game's genre
- `NA_sales` (North American sales in millions of US dollars)
- `EU_sales` (European sales in millions of US dollars)
- `JP_sales` (Japanese sales in millions of US dollars)
- `Other_sales` (Other country sales in millions of US dollars)
- `Critic_Score` (maximum of 100)
- `User_Score` (maximum of 10)
- `Rating` (ESRB)

---

## 📚 Guided Foundations (Historical Context)

The notebook `00-guided-analysis_foundations.ipynb` reflects an early stage of my data analysis learning journey, guided by TripleTen. It includes data cleaning, basic EDA, and early feature exploration, serving as a foundational block before implementing the improved structure and methodology found in the main analysis.

---

## 📂 Project Structure

```bash
├── data/
│   ├── raw/              # Original dataset(s) in CSV format
│   ├── interim/          # Intermediate cleaned versions
│   └── processed/        # Final, ready-to-analyze dataset
│
├── notebooks/
│   ├── 00-guided-analysis_foundations.ipynb     ← Initial guided project (TripleTen)
│   ├── 01_cleaning.ipynb                        ← Custom cleaning 
│   ├── 02_feature_engineering.ipynb             ← Custom feature engineering
│   ├── 03_eda_and_insights.ipynb                ← Exploratory Data Analysis & visual storytelling
│   └── 04-sda_hypotheses.ipynb                  ← Business insights and hypothesis testing
│
├── src/
│   ├── init.py              # Initialization for reusable functions
│   ├── data_cleaning.py     # Data cleaning and preprocessing functions
│   ├── data_loader.py       # Loader for raw datasets
│   ├── eda.py               # Exploratory data analysis functions
│   ├── features.py          # Creation and transformation functions for new variables to support modeling and EDA
│   └── utils.py             # General utility functions for reusable helpers
│
├── outputs/
│   └── figures/          # Generated plots and visuals
│
├── requirements/
│   └── requirements.txt      # Required Python packages
│
├── .gitignore            # Files and folders to be ignored by Git
└── README.md             # This file
```
---

🛠️ Tools & Libraries

- Python 3.11
- os, pathlib, sys, pandas, NumPy, Matplotlib, seaborn, IPython.display, scipy.stats
- Jupyter Notebook
- Git & GitHub for version control

---

## 📌 Notes

This project is part of a personal learning portfolio focused on developing strong skills in data analysis, statistical thinking, and communication of insights. Constructive feedback is welcome.

---

## 👤 Author   
##### Luis Sergio Pastrana Lemus   
##### Engineer pivoting into Data Science | Passionate about insights, structure, and solving real-world problems with data.   
##### [GitHub Profile](https://github.com/LuisPastranaLemus)   
##### 📍 Querétaro, México     
##### 📧 Contact: luis.pastrana.lemus@engineer.com   
---

