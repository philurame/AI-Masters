#!/opt/conda/envs/dsenv/bin/python3

import sys, os, logging, joblib
import pandas as pd
import numpy as np

sys.path.append('.')
from model import test_fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = joblib.load("2a.joblib")

for line in sys.stdin:
  if not line.strip(): continue

  # unpack into a df-row
  row = pd.DataFrame([line.split('\t')], columns=test_fields).map(lambda x: np.nan if x in ['\\N', '', 'NULL'] else x)

  # get probability of 1
  pred = model.predict_proba(row).reshape(-1)[1]
  print(f"{row.id[0]}\t{pred}")