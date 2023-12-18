FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

LABEL maintainer="Florian Katerndahl <florian@katerndahl.com>"
LABEL version="latest"
LABEL description="Dependency collection for tree species classification from satellite time series using neural networks"

ENV DEBIANFRONTEND=noninteractive

WORKDIR /usr/src/sits

RUN apt update && \
    apt upgrade -y && \
    apt install -y bc python3-pip jq

# courtesy of https://github.com/davidfrantz/base_image/blob/fab4748fe6d017788b7e5aa109266791838afb37/Dockerfile
RUN groupadd docker && \
	useradd -m docker -g docker -p docker && \
	chmod 0777 /home/docker && \
	chgrp docker /usr/local/bin && \
	mkdir -p /usr/scripts && \
	chown -R docker:docker /usr/src 

ENV HOME /home/docker
ENV PATH="${PATH}:/home/docker/.local/bin/"
USER docker

RUN wget -O install.py https://install.python-poetry.org && \
    python3 install.py && \
    rm install.py

# TODO why does poetry install nvidia dependency?!
COPY sits_classifier /usr/src/sits/sits_classifier
COPY .poetry/cpu-inference.toml /usr/src/sits/pyproject.toml
COPY README.md /usr/src/sits

RUN poetry install && \
    poetry build

WORKDIR ${HOME}

RUN pip install /usr/src/sits/dist/sits_classification-0.1.0-py3-none-any.whl

# TODO Nextflow allows processes to access executable scripts/binaries which are implicitly moved to the executing container/environment
#  when they're in a directory called 'bin'. Symlinked the apps directory to the bin dir. Maybe this works, dunno
#  Also not sure if this is the best idea when moving the execution to something other than local executor together with not having a 
#  bin dir in the gh-repo (which should not be created since it would duplicate files)
#COPY apps/inference.py .

CMD [ "bash" ]