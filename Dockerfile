FROM python:3.9-slim as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         libatlas-base-dev libopenblas-dev liblapack-dev \
         ca-certificates \
         build-essential \
         make \
         gcc \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 

COPY app ./opt/app
WORKDIR /opt/app

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"

RUN chmod +x train \
 && chmod +x test \
 && chmod +x tune \
 && chmod +x serve 

