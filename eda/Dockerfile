FROM jupyter/scipy-notebook

USER root

COPY requirements.txt ./
RUN pip install -r requirements.txt  && rm requirements.txt

ENV WORK_DIR ${HOME}/work