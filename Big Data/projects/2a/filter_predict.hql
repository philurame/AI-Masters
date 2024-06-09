ADD FILE projects/2a/predict.py;
ADD FILE projects/2a/model.py;
ADD FILE 2a.joblib;

CREATE TEMPORARY TABLE data_filtered AS
SELECT * FROM hw2_test WHERE if1 IS NOT NULL AND if1 > 20 AND if1 < 40;

INSERT INTO TABLE hw2_pred
SELECT TRANSFORM(*)
USING '/opt/conda/envs/dsenv/bin/python3 predict.py'
FROM data_filtered;