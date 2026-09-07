.DEFAULT_GOAL := test_pa_gluon
TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(TEST_DIR)/../build.mk
NATIVE_DIR := $(AITER_NATIVE_ROOT)/cpp_itfs/pa_gluon_aot
TORCH_PATH := $(shell $(PYTHON) -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON) -c 'import sysconfig as c; print("-L" + c.get_config_var("LIBDIR"), "-lpython" + c.get_config_var("LDVERSION"), c.get_config_var("LIBS"), c.get_config_var("SYSLIBS"))')
CPPFLAGS := -DUSE_ROCM -I$(AITER_NATIVE_ROOT)/cpp_itfs -I$(NATIVE_DIR) $(PYBIND11_INC) -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include
LDLIBS := -L$(TORCH_PATH)/lib -ltorch_python -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -Wl,-rpath,$(TORCH_PATH)/lib $(PYTHON_LDFLAGS) -lfmt -lcrypto -ldl -lpthread
HEADERS := $(wildcard $(NATIVE_DIR)/*.h $(AITER_NATIVE_ROOT)/cpp_itfs/*.h)

.PHONY: test_pa_gluon run clean
test_pa_gluon: $(BUILD_DIR)/test_pa_gluon

$(BUILD_DIR)/pa_decode_gluon_aot_lib.so: $(NATIVE_DIR)/pa_decode_gluon_aot.cpp $(HEADERS) $(TEST_DIR)/Makefile $(TEST_DIR)/../build.mk | $(BUILD_DIR)
	$(HIPCC) "$<" -o "$@" -fPIC -shared -std=c++17 -O3 $(CPPFLAGS) $(LDLIBS)

$(BUILD_DIR)/test_pa_gluon: $(TEST_DIR)/test_pa_gluon.cpp $(BUILD_DIR)/pa_decode_gluon_aot_lib.so $(HEADERS) $(TEST_DIR)/Makefile $(TEST_DIR)/../build.mk
	$(HIPCC) "$<" -L"$(BUILD_DIR)" -l:pa_decode_gluon_aot_lib.so -std=c++17 -O2 $(CPPFLAGS) $(LDLIBS) -Wl,-rpath,'$$ORIGIN' -o "$@"

run: test_pa_gluon
	cd "$(BUILD_DIR)" && env $(TEST_ENV) LD_LIBRARY_PATH="$(BUILD_DIR):$(TORCH_PATH)/lib" ./test_pa_gluon

clean:
	rm -f "$(BUILD_DIR)/test_pa_gluon" "$(BUILD_DIR)/pa_decode_gluon_aot_lib.so"
