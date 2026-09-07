# Precompiled kernels

This directory contains the code objects delivered with AITER and the selection rows used by its native assembly adapters. It is an input to the library. Builds, logs and writable caches belong elsewhere.

`manifest.json` identifies every delivered file by SHA256 and size. It also records GPU targets, the observed ELF header fields, selection-column types, and the code objects required by each selection. The runtime verifies these records before admitting a target into a separate cache; the native loader verifies the selected object's bytes again before loading it into HIP.

| Location | Contents |
| --- | --- |
| `gfx942/` | Imported code objects and selection tables for gfx942. |
| `gfx950/` | Imported code objects and selection tables for gfx950. |
| `gfx1250/` | Imported code objects and selection tables for gfx1250. |
| `manifest.json` | Versioned identity, target and selection inventory. |
| `readme.md` | Original selection-format notes, retained verbatim as historical input. |

The catalog currently contains 2,942 ELF objects and 110 selection tables. The two historical files with `.co.orig` and `.co.poc_kl_merg` suffixes are included in that object count; their presence does not make them selected implementations. No GPU qualification is implied by inclusion in this directory.

## Verify and use the resources

These commands work in a checkout and with an installed wheel:

```sh
python -m aiter.kernels verify
python -m aiter.kernels verify --target gfx950 --require-complete
python -m aiter.kernels admit --target gfx950 --cache-root /tmp/aiter-kernels
```

Verification reports both available and unavailable selections. `--require-complete` rejects a target with any declared unavailable row. Admission returns a JSON receipt identifying the original manifest, admitted directory and native index. Normal Python assembly operations perform admission when preparing their native adapter. Importing AITER, its public API or its build metadata performs no admission.

For a standalone native client, the `run` command admits the resources and passes only the resulting environment to the child process:

```sh
python -m aiter.kernels run --target gfx950 -- /absolute/path/to/native-client
```

The default snapshot lives under `AITER_JIT_DIR/kernels`, or the user's XDG AITER cache. Installed resources are never used as writable caches. `AITER_KERNELS_DIR` selects an explicit alternative **original catalog**, which must pass the same checks. The older `AITER_ASM_DIR` override is accepted as an original catalog only before admission; after admission it identifies the execution snapshot. See the [manager API](../README.md) for the distinction and receipt fields.

## What is known about the imported objects

All original objects and selection files were moved without changing their bytes. The original compiler versions, flags, assembly sources and build receipts were not provided. The manifest records that missing provenance explicitly. Its ELF checks establish the recorded code-object target and header identity; they do **not** establish an operator's argument ABI or prove numerical correctness. SHA256 checks establish integrity against the caller-selected manifest, not publisher authenticity. An alternative catalog is trusted only as far as its caller-selected origin is trusted.

There are two known unavailable selections in `gfx942/mla/mla_asm.csv`:

- `mla_pfl_qh192_vh128_m32x8_n128x1_causal0.co`
- `mla_pfl_qh192_vh128_m32x8_n128x1_causal1.co`

The imported table references them, but the repository contains neither object. The catalog retains both rows with explicit reasons. Generated dispatch headers omit these two declared unavailable rows. Unexpected missing objects remain errors. The 28 gfx942 FMHA rows with MI300 and MI308 alternatives retain both variants; the native adapter continues selecting the appropriate variant for its device.

Adding or replacing a code object requires an explicit catalog update and review. Update its hash, size and ELF identity, and review every referring selection. Keep the original build/source provenance when it is known; label it unavailable when it is not. Verification never rewrites the catalog to accept changed bytes.
