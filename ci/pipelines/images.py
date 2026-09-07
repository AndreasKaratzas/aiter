"""Compose and qualify product and framework images from the same verified wheel."""

from __future__ import annotations

from pathlib import Path

from ci.common.json import digest, require, write_json
from ci.pipelines.docker import Docker
from ci.pipelines.profile import execute_profile
from ci.qualification.catalog import load_catalog
from ci.qualification.environments import require_profile_environment
from ci.release.context import stage_context
from ci.release.images import image_record
from ci.release.recipes import load_recipes
from ci.release.wheels import load_receipt, verify_wheel


def compose_images(
    *,
    source: Path,
    controls: Path,
    wheel: Path,
    output: Path,
    role: str,
    lock: dict,
    consumers: list[dict],
    docker: Docker | None = None,
) -> dict:
    source, controls, wheel, output = (
        path.resolve() for path in (source, controls, wheel, output)
    )
    recipes = load_recipes(controls)
    require(role in recipes["product"]["targets"], "unknown product image role")
    require(not output.exists(), "image delivery output must be a new attempt")
    require(
        not output.is_relative_to(source) and not output.is_relative_to(controls),
        "image delivery must be outside checkouts",
    )
    require_profile_environment(lock, "aiter", supported=True)
    receipt = load_receipt(str(wheel) + ".receipt.json")
    verify_wheel(wheel, receipt, require_clean_source=True)
    primary = (
        "cp312" in {tag.split("-")[0] for tag in receipt.wheel.tags}
        and lock["rocm"] == "7.2"
    )
    names = []
    for consumer in consumers:
        require(
            set(consumer) == {"client", "base_image", "environment_lock", "profile"},
            "invalid declared consumer composition",
        )
        client = consumer["client"]
        require(
            client in recipes["consumers"] and client not in names,
            "unknown or duplicate consumer composition",
        )
        require(
            consumer["profile"] == recipes["consumers"][client]["profile"],
            "consumer profile differs from reviewed recipe",
        )
        require_profile_environment(
            consumer["environment_lock"], client, supported=True
        )
        require(
            consumer["base_image"] == consumer["environment_lock"]["image"],
            "consumer base differs from approved lock",
        )
        require(
            consumer["environment_lock"]["python"] == lock["python"]
            and consumer["environment_lock"]["rocm"] == lock["rocm"],
            "consumer runtime tuple differs from wheel base",
        )
        names.append(client)
    require(
        set(names) == (set(recipes["inheritance_clients"]) if primary else set()),
        "primary wheel requires every declared framework inheritance check",
    )
    output.mkdir(parents=True)
    context = output.with_name(output.name + "-context")
    stage_context(source, controls, wheel, context)
    transport = docker or Docker(output / "docker")
    tag = (
        "aiter-ci:"
        + digest(
            {"output": str(output), "wheel": receipt.artifact.to_dict(), "role": role}
        ).split(":")[1][:24]
    )
    tags = []

    def build(
        recipe: str,
        target: str | None,
        base: str,
        label: str,
        inherited: tuple[str, str] | None = None,
    ) -> dict:
        command = [
            "build",
            "-f",
            str(context / "docker" / recipe),
            "-t",
            label,
            "--build-arg",
            "BASE_IMAGE=" + base,
        ]
        if target:
            command.extend(
                [
                    "--target",
                    target,
                    "--build-arg",
                    "AITER_WHEEL=" + wheel.name,
                    "--build-arg",
                    "AITER_WHEEL_SHA256=" + receipt.artifact.sha256,
                    "--build-arg",
                    "AITER_SOURCE_REVISION=" + receipt.source.revision,
                ]
            )
        if inherited:
            parent_tag, parent_id = inherited
            require(
                transport.inspect(parent_tag)["Id"] == parent_id,
                "inherited image tag changed before composition",
            )
            # FROM resolves an image reference, not Docker's config-object ID.
            # The unique local tag is checked on both sides of its only use.
            command.extend(["--build-arg", "AITER_IMAGE=" + parent_tag])
        command.append(str(context))
        tags.append(label)
        transport.command(command)
        if inherited:
            require(
                transport.inspect(parent_tag)["Id"] == parent_id,
                "inherited image tag changed during composition",
            )
        return transport.inspect(label)

    try:
        inspection = build(recipes["product"]["dockerfile"], role, lock["image"], tag)
        write_json(output / "image.json", inspection)
        execution = inspection
        if role == "wheelhouse":
            execution = build(
                recipes["consumers"]["pytorch"]["dockerfile"],
                None,
                lock["image"],
                tag + "-pytorch",
                (tag, inspection["Id"]),
            )
        write_json(output / "execution-image.json", execution)
        execute_profile(
            source=source,
            controls=controls,
            output=output / "product",
            profile="image",
            architecture="gfx950",
            gpus="0,1",
            lock=lock,
            mode="image",
            wheel=wheel,
            image_id=execution["Id"],
            docker=transport,
        )
        inherited = []
        for consumer in consumers:
            client = consumer["client"]
            consumer_inspection = build(
                recipes["consumers"][client]["dockerfile"],
                None,
                consumer["base_image"],
                tag + "-" + client,
                (tag, inspection["Id"]),
            )
            write_json(output / (client + "-image.json"), consumer_inspection)
            execute_profile(
                source=source,
                controls=controls,
                output=output / client,
                profile=consumer["profile"],
                architecture="gfx950",
                gpus="0,1",
                lock=consumer["environment_lock"],
                mode="image",
                wheel=wheel,
                image_id=consumer_inspection["Id"],
                docker=transport,
            )
            inherited.append(
                {
                    "client": client,
                    "run": output / client / "run",
                    "base_image": consumer["base_image"],
                    "inspection": consumer_inspection,
                }
            )
        archive = output / "image.tar"
        transport.command(["save", inspection["Id"], "-o", str(archive)])
        record = image_record(
            wheel,
            output / "product/run",
            inspection,
            lock["image"],
            archive,
            catalog=load_catalog(root=controls),
            consumers=inherited,
            execution_inspection=execution,
            role=role,
        )
        write_json(output / "record.json", record)
        return record
    finally:
        # Tags are disposable aliases. Cleanup never hides failed build/test evidence.
        failures = []
        for name in tags:
            try:
                transport.command(["image", "rm", name], timeout=120)
            except (ValueError, OSError) as error:
                failures.append(str(error))
        write_json(output / "cleanup.json", {"problems": failures})
