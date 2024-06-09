#!/opt/conda/envs/dsenv/bin/python3

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

#
# Import model definition
#
from model import model, fields


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  proj_id    = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

#
# Read dataset
#
df = pd.read_csv(train_path, sep='\t', header=None, names=fields)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
  df.drop(['id', 'label'], axis=1), df['label'], test_size=0.1, random_state=42
)

#
# Train the model
#
model.fit(X_train, y_train)
logging.info(f"model score: {model.score(X_test, y_test):.3f}")

# save the model
joblib.dump(model, f"{proj_id}.joblib")