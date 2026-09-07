/* SPDX-License-Identifier: MIT */
#include <aiter/aiter.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <string.h>

int main(void)
{
    assert(aiter_abi_version() == AITER_ABI_VERSION);
    assert(strcmp(aiter_status_string(AITER_SUCCESS), "success") == 0);
    aiter_plan* plan = (aiter_plan*)1;
    assert(aiter_rmsnorm_prepare(NULL, NULL, &plan) == AITER_INVALID_ARGUMENT);
    assert(plan == NULL);
    assert(aiter_gemm_prepare(NULL, NULL, &plan) == AITER_INVALID_ARGUMENT);
    assert(aiter_execute(NULL, NULL, 0, NULL) == AITER_INVALID_ARGUMENT);
    assert(strlen(aiter_last_error()) > 0);
    aiter_plan_destroy(NULL);
    return 0;
}
