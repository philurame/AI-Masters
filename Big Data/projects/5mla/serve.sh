run_id=$1
mlflow models serve -p 7175 --env-manager=local -m mlruns/0/${run_id}/artifacts/model