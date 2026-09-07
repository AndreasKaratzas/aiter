# syntax=docker/dockerfile:1
ARG AITER_IMAGE
ARG BASE_IMAGE
FROM ${AITER_IMAGE} AS aiter
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="AITER for pytorch"
COPY --from=aiter /opt/aiter-wheel/ /opt/aiter-wheel/
COPY docker/common/wheel.py /opt/aiter-delivery/wheel.py
RUN python3 /opt/aiter-delivery/wheel.py install --wheelhouse /opt/aiter-wheel --framework pytorch
