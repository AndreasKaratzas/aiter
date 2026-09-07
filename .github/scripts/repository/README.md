# Repository maintenance scripts

This directory owns CK dependency ancestry checks, workflow job discovery, queued/running job reporting and updates to retained shard timings. These maintain the repository and its runners; they do not establish library numerical correctness.

The [script index](../README.md) links each script to its workflow caller. Monitoring and timing updates need the caller's declared GitHub credentials, repository and artifact inputs. Shared execution policy lives in [the pipeline controllers](../../../ci/pipelines/README.md).
