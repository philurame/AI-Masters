#!/opt/conda/envs/dsenv/bin/python3

def filter_cond(line_dict):
  try: 
    return 20 < int(line_dict["if1"]) < 40
  except: 
    return False
