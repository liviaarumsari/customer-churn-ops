FROM apache/spark:3.4.1

USER root

RUN pip install --no-cache-dir \
    minio \
    numpy

RUN pip install --no-cache-dir \ 
    pyarrow \
    pandas

COPY preprocess.py /opt/spark/work-dir/

CMD ["/opt/spark/bin/spark-submit", "/opt/spark/work-dir/preprocess.py"]

USER spark
