# Build candidate wheels

This package contains the procedures used by `release-build-wheels.yaml`. The workflow chooses the source, runner, container, Python matrix and artifact names. These commands install build dependencies, compile the native SDK and Python payload, repair manylinux wheels, inspect binary dependencies and record the final bytes. Publication and channel advancement are separate release operations.

The workflow checks out build controls at `github.workflow_sha` into `control/`
and the requested candidate revision into `candidate/`. All host Python commands
run from the trusted checkout with its explicit `PYTHONPATH`. Docker mounts
that checkout read-only at `/control`, the candidate at `/workspace`, and a
separate empty wheel directory at `/artifacts`. Candidate requirements and build
scripts execute inside that container. The candidate's `ci` package is never
used to run host build or receipt helpers; an older candidate does not need to
contain the current CI commands.

The legacy and manylinux paths keep their existing dependency policies: `setuptools_scm<10`, the existing requirements, Ninja and CMake. Manylinux additionally installs its chosen Torch version, auditwheel and patchelf. An explicit Torch pin/index takes precedence; otherwise the image's `rocmX.Y` tag selects the index and the default constraint is `torch<2.13`. An immutable image may retain that tag before its `@sha256:...` digest. Inputs without a usable ROCm tag must supply an explicit dependency index and still need an unambiguous build version label.

To inspect an operation without starting Docker or installing anything:

```sh
python -m ci.release.builders start --python 3.12 \
  --image rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  --control "$PWD" --source ../aiter-candidate --directory ../wheel-artifacts --dry-run

python -m ci.release.builders build --python 3.12 --flavor manylinux \
  --image pytorch/manylinux2_28-builder:rocm7.2 \
  --source ../aiter-candidate --gpu-archs 'gfx942;gfx950' \
  --version 1.0.0+20260905 --dry-run
```

Commands pass explicit argument lists to Docker, pip, CMake and auditwheel. The small `toolchain.sh` launcher sources the optional gcc-toolset environment and executes the supplied arguments. It does not construct or evaluate a build script. The native SDK build remains the qualified gfx950 subset; `GPU_ARCHS` controls the broader Python/AOT payload and does not expand that SDK claim.

`compile` runs inside the selected build container with explicit source and wheel output directories. The candidate must contain the native SDK build inputs; an older revision without that capability fails before CMake is invoked. There is no implicit legacy downgrade or native qualification claim for that revision. Its native output must be outside the checkout, and its parallelism is bounded by `MAX_JOBS` (default two). Manylinux repair excludes the same ROCm and Torch libraries that the consumer environment supplies. Original wheels remain available if repair fails. The symbol check inspects every shared-library ZIP member without extracting archive paths and rejects objdump errors or versions above the existing GLIBCXX 3.4.29 / GLIBC 2.34 ceilings; auditwheel remains responsible for the stricter manylinux platform policy.

Receipts are created only after repair, native-manifest refresh and final binary checks. Empty wheel directories fail. A dated manylinux version uses one PEP 440 local-version separator, for example `1.0.0+20260905.rocm7.2.manylinux_2_28`, rather than two plus signs.

The CPU regression tests cover isolated controller/candidate mounts, adversarial candidate Python imports, native capability rejection, argument construction, dependency selection, repair failure preservation, binary-version rejection and CLI dry runs. They do not certify a full release image build or any remote publication.
