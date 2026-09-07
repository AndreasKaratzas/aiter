# Legacy tuning-table repairs

The runtime no longer writes to source configuration files. Reading several tables either produces a content-addressed cache file or reports an ambiguous shape with its source locations. Measurements are not approved merely because their recorded time is shorter.

Enabling checks for all shipped table families exposed two existing data problems:

- Eight rows in `aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv` were one field short. They omitted `flat`, shifting the two performance columns to the left. All eight rows have `run_1stage=0`; the existing two-stage consumer sets its effective flat flag to zero. The repair inserts that explicit zero and preserves shape, kernel, timing and error values.
- `q3vl_fp4_tuned_fmoe.csv` and `qwen3_vl_fp4_tuned_fmoe.csv` overlapped on 12 workloads. The old runtime chose the lowest recorded `us`, deleted the other rows from their source file and raised an error requiring a restart. This local migration applies that same selection rule once, as a visible source change. It retains all 60 q3vl rows and the four nonoverlapping qwen3_vl rows. The 12 removed measurements remain in [retired-qwen3-vl.csv](retired-qwen3-vl.csv); [qwen3-vl-resolution.json](qwen3-vl-resolution.json) records their keys, previous source hash and selected values.

This preserves the previous resolver's intended choices. It does not establish that the recorded measurements were comparable or that those kernels meet a new correctness/performance promise. New prepared-runtime selections require the normal trial and manifest validation.

Seventeen integration cases now exercise the actual runtime configuration resolver, all registered shipped families, a planted collision and a clean merge. They verify that input hashes stay unchanged. Two MX-scale batched GEMM families intentionally have only model tables; the resolver accepts those without inventing a nonexistent canonical input. Explicitly supplied missing files still fail.
