#!/opt/conda/envs/dsenv/bin/python3

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
  ('logreg', LogisticRegression(C=0.2))
])