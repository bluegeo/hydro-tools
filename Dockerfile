# GRASS (with GDAL and Python) from the Debian archive keeps a single copy of GDAL,
# PROJ, and GEOS for GRASS and the GDAL command line tools
ARG DEBIAN_RELEASE=trixie

# Compile GRASS addons and install Python dependencies
FROM debian:${DEBIAN_RELEASE}-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    GRASS_ADDON_BASE=/opt/grass-addons

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    grass-core \
    grass-dev \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN for ext in r.stream.order r.stream.basins r.flowaccumulation; do \
    grass --tmp-project EPSG:4326 --exec g.extension extension=$ext operation=add \
    || exit 1; \
    done \
    && rm -rf ${GRASS_ADDON_BASE}/docs

COPY pyproject.toml setup.cfg README.md /hydro-tools/
COPY src/ /hydro-tools/src/

# Build rasterio and fiona against the system GDAL rather than bundling another copy
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --no-binary rasterio,fiona /hydro-tools \
    && find /opt/venv -name "*.so*" -type f -exec strip --strip-unneeded {} +


FROM debian:${DEBIAN_RELEASE}-slim

ENV DEBIAN_FRONTEND=noninteractive \
    GRASS_ADDON_BASE=/opt/grass-addons \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# Skip documentation (including the GRASS manual)
RUN printf '%s\n' \
    'path-exclude=/usr/share/doc/*' \
    'path-exclude=/usr/share/man/*' \
    'path-exclude=/usr/share/info/*' \
    'path-exclude=/usr/share/locale/*' \
    > /etc/dpkg/dpkg.cfg.d/01-nodoc

# python3-gdal provides the GDAL Python scripts (gdal_edit.py, gdal_polygonize.py).
# grass-core depends on OpenGL, which is only used by the 3D viewer (NVIZ), so the
# Mesa drivers (and LLVM) are removed after installation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gdal-bin \
    grass-core \
    python3-gdal \
    && dpkg --purge --force-depends \
    libgbm1 libglx-mesa0 libgl1-mesa-dri mesa-libgallium libllvm19 libz3-4 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/grass-addons /opt/grass-addons
COPY --from=build /opt/venv /opt/venv

ENTRYPOINT ["python3"]
