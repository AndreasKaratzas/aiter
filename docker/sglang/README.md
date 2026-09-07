# AITER in SGLang

This recipe copies the exact wheel from an AITER runtime or wheelhouse image into a compatible SGLang base. It shares `docker/common/wheel.py` with the other recipes, so checksum verification and dependency-preserving installation follow the same rules.

```bash
docker build -f docker/sglang/Dockerfile \
  --build-arg AITER_IMAGE='registry/aiter-runtime@sha256:THE_IMAGE_DIGEST' \
  --build-arg BASE_IMAGE='approved/sglang@sha256:THE_BASE_DIGEST' \
  -t aiter-sglang:candidate .
```

The pipeline executes the `sglang-image` profile inside the composed image. Its plan records the wheel hash, actual image configuration, approved environment and framework identity. These are operator and integration checks; they do not claim that every model is qualified.

The recipe does not select a floating framework version or install a replacement Torch/Triton stack. Supply the framework in its approved base image and let the qualification result show whether that exact combination works.
