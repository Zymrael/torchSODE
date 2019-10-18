# Latest as of writing
FROM nvcr.io/nvidia/pytorch:19.09-py3

COPY src /srv/src
WORKDIR /srv/src
RUN python3 setup.py install

VOLUME ["/srv"]
WORKDIR /srv/notebooks
CMD ["/bin/bash", "-c", "jupyter notebook"]
