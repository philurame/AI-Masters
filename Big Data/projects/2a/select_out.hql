INSERT OVERWRITE DIRECTORY 'philurame_hiveout'
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE
SELECT * FROM hw2_pred;