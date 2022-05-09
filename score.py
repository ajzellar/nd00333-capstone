import os
import json
import joblib
import traceback
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

# A mapping of the one hot encoded data to take a request and properly format the data to hit the model
hot_encode_map = {
  "workclass": [
    "Federal-gov",
    "Local-gov",
    "Never-worked",
    "Other",
    "Private",
    "Self-emp-inc",
    "Self-emp-not-inc",
    "State-gov",
    "Without-pay"
  ],
  "marital-status": [
    "Divorced",
    "Married-AF-spouse",
    "Married-civ-spouse",
    "Married-spouse-absent",
    "Never-married",
    "Separated",
    "Widowed"
  ],
  "occupation": [
    "Adm-clerical",
    "Armed-Forces",
    "Craft-repair",
    "Exec-managerial",
    "Farming-fishing",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Other",
    "Other-service",
    "Priv-house-serv",
    "Prof-specialty",
    "Protective-serv",
    "Sales",
    "Tech-support",
    "Transport-moving"
  ],
  "relationship": [
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Own-child",
    "Unmarried",
    "Wife"
  ],
  "race": [
    "Amer-Indian-Eskimo",
    "Asian-Pac-Islander",
    "Black",
    "Other",
    "White"
  ],
  "sex": [
    "Female",
    "Male"
  ],
  "native-country": [
    "Cambodia",
    "Canada",
    "China",
    "Columbia",
    "Cuba",
    "Dominican-Republic",
    "Ecuador",
    "El-Salvador",
    "England",
    "France",
    "Germany",
    "Greece",
    "Guatemala",
    "Haiti",
    "Holand-Netherlands",
    "Honduras",
    "Hong",
    "Hungary",
    "India",
    "Iran",
    "Ireland",
    "Italy",
    "Jamaica",
    "Japan",
    "Laos",
    "Mexico",
    "Nicaragua",
    "Other",
    "Outlying-US(Guam-USVI-etc)",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Puerto-Rico",
    "Scotland",
    "South",
    "Taiwan",
    "Thailand",
    "Trinadad&Tobago",
    "United-States",
    "Vietnam",
    "Yugoslavia"
  ]
}

input_sample = pd.DataFrame(
    {
    "age": pd.Series([0.0], dtype="int64"),
    "workclass": pd.Series([0], dtype="object"),
    "fnlwgt": pd.Series([0], dtype="int64"),
    "education": pd.Series([0], dtype="object"),
    "education-num": pd.Series([0], dtype="int64"),
    "marital-status": pd.Series([0], dtype="object"),
    "occupation": pd.Series([0.0], dtype="object"),
    "sex": pd.Series([0.0], dtype="object"),
    "race": pd.Series([0.0], dtype="object"),
    "relationship": pd.Series([0.0], dtype="object"),
    "capital-gain": pd.Series([0], dtype="int64"),
    "capital-loss": pd.Series([0], dtype="int64"),
    "hours-per-week": pd.Series([0], dtype="int64"),
    "native-country": pd.Series([0], dtype="object")
    }
)


def init():
    print(os.listdir("."))
    print(os.getenv('AZUREML_MODEL_DIR'))
    print(os.listdir(os.getenv('AZUREML_MODEL_DIR')))
    global model
    # Replace filename if needed.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

def process_input(df: DataFrame):
    if 'education' in df.columns:
        df.drop('education', axis=1, inplace=True)
    if 'salary' in df.columns:
        df.drop('salary', axis=1, inplace=True)

    num_cols = [x for x in df.columns if df[x].dtype != 'object']
    s_scaler = StandardScaler()
    # Use same scaler values as intial dataset
    mean = [38.58164675532078, 189778.36651208502, 10.0806793403151, 1077.6488437087312, 87.303829734959, 40.437455852092995]
    scale = [13.640223092304275, 105548.3568808908, 2.5726808256012865, 7385.178676947626, 402.9540308274866, 12.34723907570799]
    s_scaler.mean_ = mean
    s_scaler.scale_ = scale
    df[num_cols] = s_scaler.transform(df[num_cols])

    obj_columns = list()
    non_obj_columns = list()
    X_out = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            obj_columns.append(col)
            for sub_col in hot_encode_map[col]:
                encode_col = col + "_" + sub_col
                X_out[encode_col] = [1] if df[col][0] == sub_col else [0]
        else:
            non_obj_columns.append(col)

    X_out[non_obj_columns] = df[non_obj_columns]
    # Order columns to match training
    columns = ["workclass_Federal-gov","workclass_Local-gov","workclass_Never-worked","workclass_Other","workclass_Private","workclass_Self-emp-inc","workclass_Self-emp-not-inc","workclass_State-gov","workclass_Without-pay","marital-status_Divorced","marital-status_Married-AF-spouse","marital-status_Married-civ-spouse","marital-status_Married-spouse-absent","marital-status_Never-married","marital-status_Separated","marital-status_Widowed","occupation_Adm-clerical","occupation_Armed-Forces","occupation_Craft-repair","occupation_Exec-managerial","occupation_Farming-fishing","occupation_Handlers-cleaners","occupation_Machine-op-inspct","occupation_Other","occupation_Other-service","occupation_Priv-house-serv","occupation_Prof-specialty","occupation_Protective-serv","occupation_Sales","occupation_Tech-support","occupation_Transport-moving","relationship_Husband","relationship_Not-in-family","relationship_Other-relative","relationship_Own-child","relationship_Unmarried","relationship_Wife","race_Amer-Indian-Eskimo","race_Asian-Pac-Islander","race_Black","race_Other","race_White","sex_Female","sex_Male","native-country_Cambodia","native-country_Canada","native-country_China","native-country_Columbia","native-country_Cuba","native-country_Dominican-Republic","native-country_Ecuador","native-country_El-Salvador","native-country_England","native-country_France","native-country_Germany","native-country_Greece","native-country_Guatemala","native-country_Haiti","native-country_Holand-Netherlands","native-country_Honduras","native-country_Hong","native-country_Hungary","native-country_India","native-country_Iran","native-country_Ireland","native-country_Italy","native-country_Jamaica","native-country_Japan","native-country_Laos","native-country_Mexico","native-country_Nicaragua","native-country_Other","native-country_Outlying-US(Guam-USVI-etc)","native-country_Peru","native-country_Philippines","native-country_Poland","native-country_Portugal","native-country_Puerto-Rico","native-country_Scotland","native-country_South","native-country_Taiwan","native-country_Thailand","native-country_Trinadad&Tobago","native-country_United-States","native-country_Vietnam","native-country_Yugoslavia","age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    return X_out[columns]

def run(data):
    try:
        data_frame = pd.DataFrame(json.loads(data))
        output = model.predict(process_input(data_frame))
        return json.dumps({"result": output.tolist()})
    except Exception as e:
        traceback.print_exc()
        err = str(e)
        return json.dumps({"error": err})
