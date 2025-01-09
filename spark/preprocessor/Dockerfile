FROM apache/spark:3.4.1

USER root

RUN pip install --no-cache-dir \
    minio \
    numpy

COPY preprocess.py /opt/spark/work-dir/

COPY misc/WA_Fn-UseC_-Telco-Customer-Churn.csv /opt/spark/work-dir/dataset.csv

CMD ["/opt/spark/bin/spark-submit", "/opt/spark/work-dir/preprocess.py"]

USER spark
