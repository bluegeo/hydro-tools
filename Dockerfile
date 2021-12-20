FROM ubuntu:20.04

# System dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y wget grass grass-dev grass-doc 

# Install miniconda
ENV PATH=/root/miniconda3/bin:${PATH}
ARG PATH=/root/miniconda3/bin:${PATH}
RUN mkdir /tmp/miniconda && \
    cd /tmp/miniconda && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b && \
    rm -rf /tmp/miniconda

RUN conda update -y conda

# Python reqs
RUN conda create -y -n hydrotools python=3.9
RUN conda init bash && . ~/.bashrc && \
    conda activate hydrotools && \
    conda install -y -c conda-forge \
    numpy \
    scipy \
    numba \
    dask \
    dask-image \
    scikit-image \
    scikit-learn \
    gdal=3.3 \
    rasterio \
    pyproj && \
    pip install grass-session

# Hydrotools
RUN git clone hydrotools && \
    cd hydro-tools && \
    pip install --no-deps .