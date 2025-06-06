FROM ghcr.io/osgeo/gdal:ubuntu-small-3.11.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    manpages-dev \
    python3-pip \
    python3-venv \
    pkg-config \
    libcairo2-dev \
    libjpeg-dev \
    libgif-dev \
    grass \
    grass-doc \
    rasterio

COPY . /hydro-tools

RUN cd /hydro-tools \
    && python3 -m venv hydro-tools \
    && . hydro-tools/bin/activate \
    && pip3 install . \
    && chmod +x entrypoint.sh

ENTRYPOINT ["/hydro-tools/entrypoint.sh"]