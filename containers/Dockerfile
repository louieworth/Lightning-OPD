ARG BASE_IMAGE=tonyhao96/jetmoe@sha256:d79db3699f58172088c8897efd4e21aa7960cbd5530c81b02a6bf07132695313
FROM ${BASE_IMAGE}
WORKDIR /opt/Lightning-OPD
COPY . /opt/Lightning-OPD
RUN bash containers/install.sh
ENV MEGATRON_PATH=/opt/Megatron-LM \
    CUDA_HOME=/usr/local/cuda \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/bin/bash", "/opt/Lightning-OPD/containers/entrypoint.sh"]
CMD ["bash"]
