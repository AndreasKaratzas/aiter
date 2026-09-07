# Build, verify and deliver artifacts

Release workflows connect candidate construction, qualification and publication. Start with [the release process](../../../ci/release/README.md) before changing a promotion rule; a passing build alone does not make an artifact supported.

- [build-wheels.yaml](build-wheels.yaml) builds the declared wheel matrix; [triton-wheel.yaml](triton-wheel.yaml) supplies the selected Triton wheel.
- [wheel-smoke.yaml](wheel-smoke.yaml) checks installed wheels. [images.yaml](images.yaml) composes and checks the declared delivery images.
- [nightly.yaml](nightly.yaml) and [stable.yaml](stable.yaml) orchestrate their separate delivery paths.
- [promote.yaml](promote.yaml) promotes verified artifacts; [channels.yaml](channels.yaml) records outcomes and the last qualified artifacts, retaining failed history.

The [channel guide](../../../ci/release/operations.md) explains immutable artifact identities, qualification inputs and publication permissions. The [container guide](../../../docker/README.md) explains wheel consumption by PyTorch, vLLM and SGLang. A rolling framework canary is separate from supported locked qualification.

Recurring invocations are in [release schedules](../schedules/release/README.md). Their existence does not establish that registry credentials, GPU executors or publication have been activated. Local acceptance and remaining execution requirements are recorded separately.

Return to the [workflow source guide](../README.md).
