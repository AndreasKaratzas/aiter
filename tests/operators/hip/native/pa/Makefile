.DEFAULT_GOAL := pa_ragged_test
TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(TEST_DIR)/../build.mk
NATIVE_DIR := $(AITER_NATIVE_ROOT)/cpp_itfs/pa
HEADERS := $(wildcard $(NATIVE_DIR)/*.h $(NATIVE_DIR)/*.cuh $(AITER_NATIVE_ROOT)/cpp_itfs/*.h)

.PHONY: pa_ragged_test run clean
pa_ragged_test: $(BUILD_DIR)/pa_ragged_test

$(BUILD_DIR)/pa_ragged_lib.so: $(NATIVE_DIR)/pa_ragged.cpp $(HEADERS) $(TEST_DIR)/Makefile $(TEST_DIR)/../build.mk | $(BUILD_DIR)
	$(HIPCC) "$<" -o "$@" -fPIC -shared -std=c++17 -O3 -lfmt -lcrypto -ldl

$(BUILD_DIR)/pa_ragged_test: $(TEST_DIR)/pa_ragged_test.cpp $(BUILD_DIR)/pa_ragged_lib.so $(HEADERS) $(TEST_DIR)/Makefile $(TEST_DIR)/../build.mk
	$(HIPCC) "$<" -L"$(BUILD_DIR)" -l:pa_ragged_lib.so -I"$(NATIVE_DIR)" -std=c++17 -O2 -lpthread -lfmt -lcrypto -Wl,-rpath,'$$ORIGIN' -o "$@"

run: pa_ragged_test
	cd "$(BUILD_DIR)" && env $(TEST_ENV) LD_LIBRARY_PATH="$(BUILD_DIR)" ./pa_ragged_test

clean:
	rm -f "$(BUILD_DIR)/pa_ragged_test" "$(BUILD_DIR)/pa_ragged_lib.so"
