"""A failed or stale remote write must never become an unconditional overwrite."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ci.common.json import digest, write_json
from ci.release.channels import channel_index, read_state
from ci.release.storage import fetch, publish


class StorageBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Path(self.temporary.name) / "store"
        self.output = Path(self.temporary.name) / "views"

    def test_absent_state_is_distinct_from_authentication_failure(self):
        for reason, accepted in (("(NoSuchKey)", True), ("(AccessDenied)", False)):
            with patch(
                "ci.release.storage._aws",
                return_value=SimpleNamespace(returncode=1, stderr=reason),
            ):
                if accepted:
                    self.assertIsNone(
                        fetch(self.store, "bucket", "channels/state.json")["etag"]
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "cannot read"):
                        fetch(self.store, "bucket", "channels/state.json")

    def test_compare_and_swap_never_retries_as_an_unconditional_write(self):
        self.store.mkdir()
        state = read_state(self.store)
        write_json(self.store / "state.json", state)
        write_json(
            self.store / "remote.json",
            {
                "bucket": "bucket",
                "key": "channels/state.json",
                "etag": '"prior"',
                "state_digest": state["state_digest"],
            },
        )
        write_json(self.output / "index.json", channel_index(self.store))
        write_json(self.output / "metrics.json", {"metrics_digest": digest({})})
        responses = [SimpleNamespace(returncode=0, stdout="{}", stderr="")] * 2 + [
            SimpleNamespace(returncode=1, stderr="(PreconditionFailed)")
        ]
        with patch(
            "ci.release.storage._aws", side_effect=responses
        ) as client, self.assertRaisesRegex(ValueError, "refetch and reconstruct"):
            publish(self.store, self.output)
        self.assertEqual(client.call_count, 3)
        self.assertEqual(client.call_args.args[-2:], ("--if-match", '"prior"'))

    def test_self_consistent_state_digest_cannot_hide_a_modified_index(self):
        self.store.mkdir()
        write_json(
            self.store / "remote.json",
            {"bucket": "bucket", "key": "channels/state.json", "etag": None},
        )
        index = channel_index(self.store)
        index["channels"] = {"forged": {}}
        write_json(self.output / "index.json", index)
        with patch("ci.release.storage._aws") as client, self.assertRaisesRegex(
            ValueError, "index does not describe"
        ):
            publish(self.store, self.output)
        client.assert_not_called()
