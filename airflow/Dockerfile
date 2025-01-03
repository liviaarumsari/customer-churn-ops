FROM apache/airflow:2.6.3

ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=$AIRFLOW_HOME/dags

RUN pip install --no-cache-dir \
    apache-airflow-providers-cncf-kubernetes \
    minio

USER root
RUN mkdir -p $AIRFLOW_HOME && \
    chown -R airflow:root $AIRFLOW_HOME

COPY main.py $AIRFLOW_HOME/dags/
RUN chown airflow:root $AIRFLOW_HOME/dags/main.py

USER airflow