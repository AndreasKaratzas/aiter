# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS wheel
ARG AITER_WHEEL
ARG AITER_WHEEL_SHA256
WORKDIR /opt/aiter-delivery
COPY docker/common/wheel.py /opt/aiter-delivery/wheel.py
COPY dist/ /opt/aiter-delivery/dist/
RUN python3 /opt/aiter-delivery/wheel.py stage --source dist \
    --wheel "$AITER_WHEEL" --sha256 "$AITER_WHEEL_SHA256" --destination /opt/aiter-wheel

FROM ${BASE_IMAGE} AS runtime
ARG AITER_SOURCE_REVISION
LABEL org.opencontainers.image.title="AITER runtime" \
      org.opencontainers.image.source="https://github.com/ROCm/aiter" \
      org.opencontainers.image.revision="$AITER_SOURCE_REVISION"
COPY --from=wheel /opt/aiter-wheel /opt/aiter-wheel
COPY --from=wheel /opt/aiter-delivery/wheel.py /opt/aiter-delivery/wheel.py
RUN python3 /opt/aiter-delivery/wheel.py install --wheelhouse /opt/aiter-wheel --framework pytorch
WORKDIR /opt/aiter

FROM runtime AS development
LABEL org.opencontainers.image.title="AITER development"
RUN c++ --version && /opt/rocm/lib/llvm/bin/clang++ --version \
    && (command -v ninja || command -v make) \
    && python3 -c 'import re, subprocess, sys; output = subprocess.check_output(["cmake", "--version"], text=True); version = tuple(map(int, re.search(r"version (\d+)\.(\d+)", output).groups())); sys.exit(0 if version >= (3, 21) else "AITER development requires CMake >=3.21 in the approved base")'
COPY include/ /opt/aiter/include/
COPY ci/ /opt/aiter/ci/
COPY tests/ /opt/aiter/tests/

FROM scratch AS wheelhouse
COPY --from=wheel /opt/aiter-wheel /opt/aiter-wheel
