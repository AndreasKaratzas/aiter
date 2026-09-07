# Experimental code-object inspection helpers

These utilities support [native code-object inspection](../../isa_kernel_optimization.md). They are development experiments, outside the product qualification profiles and installed runtime.

| Helper | Purpose |
|---|---|
| [extract_asm.py](extract_asm.py) | Inspect a code object or emit a reconstructed assembly file. |
| [analyze_kernel.py](analyze_kernel.py) | Summarize instructions or read an existing profiler result. |
| [roundtrip.sh](roundtrip.sh) | Reassemble into temporary output and compare selected ELF sections. |
| [Dockerfile](Dockerfile) | Historical experimental tool environment; not a qualified AITER image. |

From the repository root, a read-only inspection is:

```bash
python docs/examples/isa_optimization/extract_asm.py \
  kernels/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co --list
python docs/examples/isa_optimization/analyze_kernel.py isa \
  kernels/gfx942/pa/pa_bf16_pertokenFp8_gqa16_2tg_4w.co
```

The tools use the ROCm LLVM installation selected by their arguments or environment. Reassembly may differ from the original encoding. A matching section comparison is not a numerical, concurrency or performance test. Keep generated artifacts outside the checkout and use a new qualification record for any proposed replacement.
