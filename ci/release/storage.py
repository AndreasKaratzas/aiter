"""Move reviewed channel records through an S3 compare-and-swap boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.release.channels import channel_index, read_state


def _aws(*arguments: str):
    return subprocess.run(
        ["aws", "s3api", *arguments, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )


def fetch(store: Path, bucket: str, key: str) -> dict:
    require(
        bucket and key and not key.startswith("/"),
        "configure a bucket and relative state key",
    )
    store.mkdir(parents=True, exist_ok=True)
    target = store / "state.json"
    require(not target.exists(), "fetch requires a new local state directory")
    result = _aws("get-object", "--bucket", bucket, "--key", key, str(target))
    if result.returncode:
        require(
            "(NoSuchKey)" in result.stderr,
            "cannot read channel state: " + result.stderr,
        )
        target.unlink(missing_ok=True)
        etag = None
    else:
        etag = json.loads(result.stdout)["ETag"]
        require(isinstance(etag, str) and etag, "remote state has no ETag")
    remote = {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "state_digest": read_state(store)["state_digest"],
    }
    write_json(store / "remote.json", remote)
    return remote


def publish(store: Path, output: Path) -> dict:
    remote = load_json(store / "remote.json")
    state = read_state(store)
    # Derived views are immutable; readers use the digest in the authoritative state.
    index = load_json(output / "index.json")
    require(
        index == channel_index(store, as_of=index.get("as_of_utc")),
        "index does not describe this channel state",
    )
    metrics = load_json(output / "metrics.json")
    require(
        metrics.get("metrics_digest")
        == digest(
            {key: value for key, value in metrics.items() if key != "metrics_digest"}
        ),
        "metrics summary digest mismatch",
    )
    identity = state["state_digest"].split(":", 1)[1]
    for name in ("index.json", "metrics.json"):
        path = output / name
        require(path.is_file(), "missing channel view " + name)
        result = _aws(
            "put-object",
            "--bucket",
            remote["bucket"],
            "--key",
            remote["key"] + ".views/" + identity + "/" + name,
            "--body",
            str(path),
            "--content-type",
            "application/json",
            "--if-none-match",
            "*",
        )
        if result.returncode and "(PreconditionFailed)" in result.stderr:
            existing = output / (name + ".remote")
            fetched = _aws(
                "get-object",
                "--bucket",
                remote["bucket"],
                "--key",
                remote["key"] + ".views/" + identity + "/" + name,
                str(existing),
            )
            require(
                fetched.returncode == 0 and existing.read_bytes() == path.read_bytes(),
                "existing immutable channel view differs",
            )
            existing.unlink()
        else:
            require(
                result.returncode == 0, "cannot retain channel view: " + result.stderr
            )
    condition = (
        ["--if-match", remote["etag"]] if remote["etag"] else ["--if-none-match", "*"]
    )
    result = _aws(
        "put-object",
        "--bucket",
        remote["bucket"],
        "--key",
        remote["key"],
        "--body",
        str(store / "state.json"),
        "--content-type",
        "application/json",
        *condition,
    )
    require(
        result.returncode == 0,
        "channel changed or publication failed; refetch and reconstruct rather than overwriting: "
        + result.stderr,
    )
    return {
        "state_digest": state["state_digest"],
        "etag": json.loads(result.stdout)["ETag"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch")
    get.add_argument("--bucket", required=True)
    get.add_argument("--key", required=True)
    put = commands.add_parser("publish")
    put.add_argument("--output", type=Path, required=True)
    for command in (get, put):
        command.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    result = (
        fetch(args.store, args.bucket, args.key)
        if args.command == "fetch"
        else publish(args.store, args.output)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
