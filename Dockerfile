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
    grass-dev \
    rasterio

# Install GRASS Addons
RUN grass --tmp-location EPSG:4326 --exec g.extension -s extension=r.stream.order operation=add

COPY src/ /hydro-tools/src/
COPY entrypoint.sh /hydro-tools/entrypoint.sh
COPY pyproject.toml /hydro-tools/pyproject.toml
COPY setup.cfg /hydro-tools/setup.cfg

RUN cd /hydro-tools \
    && python3 -m venv hydro-tools \
    && . hydro-tools/bin/activate \
    && pip3 install . \
    && chmod +x entrypoint.sh

ENTRYPOINT ["/hydro-tools/entrypoint.sh"]