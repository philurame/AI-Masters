#!/opt/conda/envs/dsenv/bin/python3

import sys
import os
from glob import glob
import logging

sys.path.append('.')
from model import test_fields
from filter_cond import filter_cond

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# logging.info(f"FILTERS {path_filter_cond}")  

for line in sys.stdin:
  # skip header
  if line.startswith(test_fields[0]): continue

  #unpack into a tuple/dict
  data_row_dict = dict(zip(test_fields, line.split('\t')))

  #apply filter conditions
  if filter_cond(data_row_dict): print(line)