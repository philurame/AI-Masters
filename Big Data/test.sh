files=projects/1a/filter.py,projects/1a/model.py,projects/1a/filter_cond.py,1a.joblib,projects/1a/predict.py
data=/datasets/criteo/testdir50/test-with-id-50.txt
outp=predicts.csv
mapper=filter.py
reducer=predict.py

hdfs dfs -rm -r -f -skipTrash $outp
projects/1a/filter_predict.sh $files $data $outp $mapper $reducer