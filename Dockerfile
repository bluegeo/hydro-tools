FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
    git \
    wget \
    ca-certificates \
    build-essential \
    libblas-dev \
    libbz2-dev \
    libcairo2-dev \
    libfftw3-dev \
    libfreetype6-dev \
    libglu1-mesa-dev \
    libgsl0-dev \
    libjpeg-dev \
    liblapack-dev \
    liblas-dev \
    liblas-c-dev \
    libncurses5-dev \
    libnetcdf-dev \
    libopenjp2-7 \
    libopenjp2-7-dev \
    libpdal-dev pdal \
    libpdal-plugin-python \
    libpng-dev \
    libpq-dev \
    libreadline-dev \
    libsqlite3-dev \
    libtiff-dev \
    libxmu-dev \
    libzstd-dev \
    bison \
    flex \
    g++ \
    gettext \
    libfftw3-bin \
    make \
    ncurses-bin \
    netcdf-bin \
    sqlite3 \
    subversion \
    unixodbc-dev \
    zlib1g-dev && \
    apt-get autoremove && \
    apt-get clean

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

# GRASS
ENV MYCFLAGS "-O2 -std=gnu99 -m64"
ENV MYLDFLAGS "-s"
ENV LD_LIBRARY_PATH "/usr/local/lib"
ENV LDFLAGS "$MYLDFLAGS"
ENV CFLAGS "$MYCFLAGS"
ENV CXXFLAGS "$MYCXXFLAGS"

RUN conda init bash && . ~/.bashrc && \
    conda activate hydrotools && \
    git clone -b releasebranch_7_8 https://github.com/OSGeo/grass.git && \
    cd grass && \
    ./configure \
    --enable-largefile \
    --with-cxx \
    --with-nls \
    --with-readline \
    --with-sqlite \
    --with-bzlib \
    --with-zstd \
    --with-cairo --with-cairo-ldflags=-lfontconfig \
    --with-freetype --with-freetype-includes="/usr/include/freetype2/" \
    --with-fftw \
    --with-netcdf \
    --with-liblas --with-liblas-config=/usr/bin/liblas-config \
    --with-gdal=/root/miniconda3/envs/hydrotools/bin/gdal-config \
    --with-proj --with-proj-share=/root/miniconda3/envs/hydrotools/share/proj \
    --with-geos=/root/miniconda3/envs/hydrotools/bin/geos-config \
    --with-postgres --with-postgres-includes="/usr/include/postgresql" \
    --with-opengl-libs=/usr/include/GL && \
    make && \
    make install && \
    ldconfig && \
    rm -rf ../grass

# Hydrotools
# COPY pyproject.toml /tmp/hydro-tools/
# COPY setup.cfg /tmp/hydro-tools/
# COPY src /tmp/hydro-tools/src

# RUN conda init bash && . ~/.bashrc && \
#     conda activate hydrotools && \
#     cd /tmp/hydrotools && \
#     pip install --no-deps . && \
#     rm -rf /tmp/hydro-tools