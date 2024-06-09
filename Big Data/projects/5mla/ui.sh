export MLFLOW_TRACKING_URI="http://localhost:6175"
mlflow ui --port 6175 --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root file:./mlruns