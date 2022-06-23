FROM osgeo/gdal:ubuntu-full-latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    manpages-dev \
    python3-pip \
    pkg-config \
    libcairo2-dev \
    libjpeg-dev \
    libgif-dev \
    grass \
    grass-doc \
    rasterio

COPY . /hydro-tools

RUN cd /hydro-tools && pip3 install .