import os
import joblib
from azureml.core.run import Run
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from azureml.data.dataset_factory import TabularDatasetFactory
import argparse


def prep_data(df : DataFrame):
  # Replace missing values from dataset
  df.replace(' ?', np.NaN, inplace=True)
  df.isna().sum()
  for col in df:
    df[col].fillna(value="Other", inplace=True)

  # Get column to predict
  y_out = df['salary'].map( {' >50K': 1, ' <=50K': 0} )
  df.drop('salary', axis=1, inplace=True)

  # This field isn't required as we already have the education-num field which assigns numerical value for education
  df.drop('education', axis=1, inplace=True)

  # standardize numerical features
  num_cols = [x for x in df.columns if df[x].dtype != 'object']
  s_scaler = StandardScaler()
  s_scaler.fit(df[num_cols])

  # Print mean and scaler for use in inference
  mean_list = [float(x) for x in s_scaler.mean_]
  scale_list = [float(x) for x in s_scaler.scale_]
  print(mean_list)
  print(scale_list)
  df[num_cols] = s_scaler.transform(df[num_cols])

  # Find category type columns
  obj_columns = list()
  non_obj_columns = list()
  for col in df.columns:
    if df[col].dtype == 'object':
      df[col] = df[col].str.strip()
      obj_columns.append(col)
    else:
      non_obj_columns.append(col)
  # One Hot Encode categories
  X_out = pd.get_dummies(df[obj_columns])
  X_out[non_obj_columns] = df[non_obj_columns]
  # print(X_out.head())
  # print(X_out.info())
  return X_out, y_out

def get_dataset():
  github_path = "https://raw.githubusercontent.com/ajzellar/nd00333-capstone/master/salary.csv"
  dataset = TabularDatasetFactory.from_delimited_files(path=github_path)
  return dataset.to_pandas_dataframe()


def main():
  # Add arguments to script
  parser = argparse.ArgumentParser()
  parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
  parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
  args = parser.parse_args()

  run = Run.get_context()
  run.log("Regularization Strength:", np.float64(args.C))
  run.log("Max iterations:", np.int(args.max_iter))

  df = get_dataset()
  X, y = prep_data(df)
  X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)

  model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(X_train, y_train)
  accuracy = model.score(X_test, y_test)
  run.log("Accuracy", np.float(accuracy))

  outputs_dir = 'outputs'
  os.makedirs(outputs_dir, exist_ok=True)
  output_model = os.path.join(outputs_dir, "model.joblib".format(run.id))
  joblib.dump(model, output_model)

if __name__ == '__main__':
    main()