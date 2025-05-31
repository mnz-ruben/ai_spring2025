#!/usr/bin/env python3
"""
regression_analysis.py

Review the relationship between “age” and “bmi”, and between “age” and “charges”
using simple linear regression on the insurance dataset.

Make sure that:
  • You have a file named `insurance.csv` in the same folder as this script.
  • You have installed pandas and statsmodels in your virtual environment:
      pip install pandas statsmodels
"""

import os
import pandas as pd
import statsmodels.api as sm

def main():
    # Construct the path to insurance.csv located in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'insurance.csv')

    # Load the insurance dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find 'insurance.csv' in:\n  {csv_path}")
        print("Please make sure 'insurance.csv' is in the same folder as this script.")
        return

    # ---- Regression 1: BMI ~ Age ----
    X1 = sm.add_constant(df['age'])  # Adds the intercept term
    y1 = df['bmi']
    model1 = sm.OLS(y1, X1).fit()

    # ---- Regression 2: Charges ~ Age ----
    X2 = sm.add_constant(df['age'])
    y2 = df['charges']
    model2 = sm.OLS(y2, X2).fit()

    # Print summaries
    print("==============================================")
    print("Regression 1:   BMI ~ Age")
    print("----------------------------------------------")
    print(model1.summary())
    print("\n\n")
    print("==============================================")
    print("Regression 2:   Charges ~ Age")
    print("----------------------------------------------")
    print(model2.summary())

if __name__ == "__main__":
    main()
