FROM ubuntu

MAINTAINER Christopher Dembia

# Build this docker container with a command like the following:
#
#   docker build --tag <username>/mocopaper:<tag> .
#
# When building the Docker container on Windows or Mac, make sure Docker has
# access to at least 8 GB of RAM.
# https://stackoverflow.com/questions/44533319/how-to-assign-more-memory-to-docker-container

# Set DEBIAN_FRONTEND to avoid interactive timezone prompt when installing
# packages.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        build-essential libtool autoconf \
        cmake \
        gfortran \
        wget \
        pkg-config \
        libopenblas-dev \
        liblapack-dev \
        swig \
        python3 python3-dev python3-numpy python3-setuptools

# Must be careful to not embed the GitHub token in the image.
RUN git clone https://github.com/opensim-org/opensim-moco.git /opensim-moco \
        && cd /opensim-moco \
        && git checkout master

RUN cd /opensim-moco \
        && git submodule update --init \
        && mkdir ../moco_dependencies_build \
        && cd ../moco_dependencies_build \
        && cmake ../opensim-moco/dependencies -DOPENSIM_PYTHON_WRAPPING=on \
        && make --jobs $(nproc) ipopt \
        && make --jobs $(nproc) \
        && echo "/moco_dependencies_install/adol-c/lib64" >> /etc/ld.so.conf.d/moco.conf \
        && echo "/moco_dependencies_install/ipopt/lib" >> /etc/ld.so.conf.d/moco.conf \
        && ldconfig \
        && rm -r /moco_dependencies_build

RUN cd / \
        && mkdir build \
        && cd build \
        && cmake ../opensim-moco \
            -DMOCO_PYTHON_BINDINGS=on \
            -DBUILD_TESTING=off \
            -DBUILD_EXAMPLES=off \
        && make --jobs $(nproc) install \
        && echo "/opensim-moco-install/sdk/lib" >> /etc/ld.so.conf.d/moco.conf \
        && echo "/opensim-moco-install/sdk/Simbody/lib" >> /etc/ld.so.conf.d/moco.conf \
        && ldconfig \
        && cd /opensim-moco-install/sdk/Python && python3 setup.py install \
        && rm -r /build

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula \
        select true | debconf-set-selections

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3-pip python3-scipy python3-opencv \
        ttf-mscorefonts-installer \
        texlive-full rubber


# We need matplotlib 3.1.
RUN pip3 install matplotlib

# Mount a volume to the Docker container's /output folder to save outputs
# to the local machine.
RUN echo "Creating the output directory." && mkdir /output

RUN git clone https://github.com/stanfordnmbl/mocopaper /mocopaper

# Matplotlib's default backend requires a DISPLAY / Xserver.
# RUN echo 'backend : Agg' >> /mocopaper/matplotlibrc && \
#     echo 'font.sans-serif : Arial, Helvetica, sans-serif' >> /mocopaper/matplotlibrc

WORKDIR /mocopaper

ENTRYPOINT ["/bin/bash", "mocopaper.sh"]

