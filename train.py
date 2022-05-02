import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse


def prep_data(df : DataFrame):
  # Replace missing values from dataset
  df.replace(' ?', np.NaN, inplace=True)
  df.isna().sum()
  for col in df:
    df[col].fillna(value="Other", inplace=True)

  # Get column to predict
  y = df['salary'].map( lambda s: 1 if s == " >50K" else 0 )
  df.drop('salary', axis=1, inplace=True)

  # This field isn't required as we already have the education-num field which assigns numerical value for education
  df.drop('education', axis=1, inplace=True)

  # standardize numerical features
  num_cols = [x for x in df.columns if df[x].dtype != 'object']
  df[num_cols] = StandardScaler().fit_transform(df[num_cols])

  # Find category type columns
  obj_columns = list()
  for col in df.columns:
    if df[col].dtype == 'object':
      df[col] = df[col].str.strip()
      obj_columns.append(col)

  df = pd.get_dummies(df, columns=obj_columns, drop_first=True)

  return df, y

def main():
  # Add arguments to script
  parser = argparse.ArgumentParser()
  parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
  parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
  args = parser.parse_args()


if __name__ == '__main__':
    main()