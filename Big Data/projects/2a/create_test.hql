CREATE TEMPORARY EXTERNAL TABLE hw2_test (
    id INT,
    if1 DOUBLE, if2 DOUBLE, if3 DOUBLE, if4 DOUBLE, if5 DOUBLE, if6 DOUBLE, if7 DOUBLE, if8 DOUBLE, if9 DOUBLE, if10 DOUBLE, if11 DOUBLE, if12 DOUBLE, if13 DOUBLE,
    cf1 STRING, cf2 STRING, cf3 STRING, cf4 STRING, cf5 STRING, cf6 STRING, cf7 STRING, cf8 STRING, cf9 STRING, cf10 STRING, cf11 STRING, cf12 STRING, cf13 STRING,
    cf14 STRING, cf15 STRING, cf16 STRING, cf17 STRING, cf18 STRING, cf19 STRING, cf20 STRING, cf21 STRING, cf22 STRING, cf23 STRING, cf24 STRING, cf25 STRING, cf26 STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LOCATION '/datasets/criteo/testdir';