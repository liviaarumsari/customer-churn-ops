FROM apache/spark:3.4.1
COPY preprocess.py /opt/spark/work-dir/

# placeholder for data source
# TODO: replace with fetch dataset
COPY ../../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv /opt/spark/work-dir/dataset.csv

CMD ["/opt/spark/bin/spark-submit", "/opt/spark/work-dir/preprocess.py"]
