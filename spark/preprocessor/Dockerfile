FROM apache/spark:3.4.1
COPY preprocess.py /opt/spark/work-dir/
CMD ["/opt/spark/bin/spark-submit", "/opt/spark/work-dir/preprocess.py"]
