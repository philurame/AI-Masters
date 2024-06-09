#!/opt/conda/envs/dsenv/bin/python3

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

import mlflow

train_path = sys.argv[1]
model_param1 = float(sys.argv[2])

# =====================================================================================
# model definition

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]
fields = ["id", "label"] + numeric_features + categorical_features
test_fields =     ["id"] + numeric_features + categorical_features
categorical_features = ['cf5', 'cf6', 'cf8', 'cf9', 'cf14', 'cf17', 'cf20', 'cf22', 'cf23', 'cf25']


numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
  ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
  ('num', numeric_transformer, numeric_features),
  ('cat', categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('logreg', LogisticRegression(C=model_param1))
])

# =====================================================================================

df = pd.read_csv(train_path, sep='\t', header=None, names=fields)

X_train, X_test, y_train, y_test = train_test_split(
  df.drop(['id', 'label'], axis=1), df['label'], test_size=0.1, random_state=42
)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

mlflow.log_param("model_param1", model_param1)
mlflow.log_metric("log_loss", score)
mlflow.sklearn.log_model(model, artifact_path="model")