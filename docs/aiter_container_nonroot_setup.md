# Run a container without root privileges

Start with a wheel and image whose Python, ROCm, Torch and GPU target match your workload. The [container guide](../docker/README.md) explains the common image and framework-specific consumers. A floating `latest` tag does not identify the environment that produced a qualification result.

Running the application as a non-root user has two practical requirements: access to the GPU device nodes, and a writable place for permitted compilation caches. It does not require reinstalling AITER into the source tree.

## Match the host's device groups

Inspect the device permissions on the host:

```bash
ls -l /dev/kfd /dev/dri/renderD*
id
```

Pass the numeric group IDs that own those devices to Docker with `--group-add`. Group names inside an image need not match the host names. Run with your intended UID and GID, pass `/dev/kfd` and `/dev/dri`, and mount an application-owned cache directory if the workload permits compilation.

For example, after setting `AITER_IMAGE` to the reviewed image digest and `KFD_GROUP` and `RENDER_GROUP` to the observed numeric group IDs:

```bash
mkdir -p /tmp/aiter-user-cache
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$KFD_GROUP" --group-add "$RENDER_GROUP" \
  --mount type=bind,source=/tmp/aiter-user-cache,target=/cache \
  --env XDG_CACHE_HOME=/cache \
  "$AITER_IMAGE" bash
```

Inside the container, verify `id`, device permissions and `rocminfo`, then run the small operator check appropriate to the image's qualification profile. Device visibility alone does not prove numerical correctness or artifact compatibility.

## Keep runtime and development roles explicit

A development image may include a compiler and permit new kernel builds. A runtime consumer that requires prebuilt code must use the corresponding artifact policy and qualified image. An unwritable cache should not be fixed by making the installed package writable; use an application-owned cache or choose the matching prebuilt artifact.

The documentation build does not execute Docker. Actual image qualification and its evidence are owned by the [delivery pipeline](../ci/README.md).
