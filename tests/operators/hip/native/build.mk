# Shared, explicit locations for native operator test builds.
AITER_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/../../../..)
PYTHON ?= python3
HIPCC ?= hipcc
ifeq ($(strip $(BUILD_DIR)),)
$(error Specify BUILD_DIR=/absolute/writable/directory outside the source checkout)
endif
ifeq ($(filter /%,$(BUILD_DIR)),)
$(error BUILD_DIR must be absolute)
endif
override BUILD_DIR := $(shell $(PYTHON) -c 'from pathlib import Path; print(Path("$(BUILD_DIR)").resolve())')
ifneq ($(filter $(AITER_ROOT) $(AITER_ROOT)/%,$(BUILD_DIR)),)
$(error BUILD_DIR must be outside the source checkout)
endif
AITER_PYTHON_ROOT ?= $(AITER_ROOT)
AITER_NATIVE_ROOT ?= $(AITER_ROOT)/csrc
PYTHON_SITES := $(shell $(PYTHON) -c 'import site; print(":".join(site.getsitepackages()))')
PYTHON_BIN := $(dir $(shell command -v $(PYTHON)))
TEST_ENV = PYTHONPATH="$(AITER_PYTHON_ROOT):$(PYTHON_SITES)" AITER_AOT_CACHE_DIR="$(BUILD_DIR)/kernels" AITER_JIT_DIR="$(BUILD_DIR)/jit" PATH="$(PYTHON_BIN):$(PATH)"

$(BUILD_DIR):
	mkdir -p "$@"
