export MLFLOW_TRACKING_URI="http://localhost:6175"
mlflow run . -P train_path=/home/users/datasets/criteo/train1000.txt --env-manager=local