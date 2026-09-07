"""Check GitHub's documented ownership matching and last-rule behavior."""

import unittest

from ci.ownership.policy import (
    effective_owner,
    load_policy,
    parse_codeowners,
    render_codeowners,
)


class CodeOwnerTests(unittest.TestCase):
    def owner(self, pattern, path):
        rules = parse_codeowners("* @fallback\n" + pattern + " @specific\n")
        return effective_owner(rules, path).owners

    def test_github_directory_and_star_examples(self):
        cases = [
            ("/docs/", "docs/build/troubleshooting.md", True),
            ("/docs/", "nested/docs/a.md", False),
            ("docs/*", "docs/start.md", True),
            ("docs/*", "docs/build/troubleshooting.md", False),
            ("apps/", "deep/apps/project/file.py", True),
            ("**/logs", "build/logs/debug.txt", True),
            ("*.py", "deep/module.py", True),
            ("/src/**/kernel?.cu", "src/kernels/device/kernel1.cu", True),
            ("/src/**/kernel?.cu", "src/kernel1.cu", True),
            ("/src/**/kernel?.cu", "src/kernel12.cu", False),
            ("/src/", "Src/file.py", False),
        ]
        for pattern, path, matched in cases:
            with self.subTest(pattern=pattern, path=path):
                self.assertEqual(
                    self.owner(pattern, path),
                    ("@specific",) if matched else ("@fallback",),
                )

    def test_last_match_replaces_instead_of_accumulating(self):
        rules = parse_codeowners(
            "* @first\n/ci/ @second @third\n/ci/ownership/owners.json @last\n"
        )
        self.assertEqual(
            effective_owner(rules, "ci/run.py").owners, ("@second", "@third")
        )
        self.assertEqual(
            effective_owner(rules, "ci/ownership/owners.json").owners, ("@last",)
        )

    def test_explicit_owner_removal_and_inline_comments(self):
        rules = parse_codeowners("/apps/ @owner # comment\n/apps/github\n")
        self.assertEqual(effective_owner(rules, "apps/github/file.py").owners, ())
        self.assertIsNone(effective_owner(rules, "other/file.py"))

    def test_unsupported_gitignore_constructs_are_rejected(self):
        for pattern in ("!secret", "[ab].py", r"\#name"):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                parse_codeowners(pattern + " @owner")

    def test_only_actual_local_steward_is_assigned_until_acceptance(self):
        policy = load_policy()
        rules = parse_codeowners(render_codeowners(policy))
        self.assertTrue(all(rule.owners == ("@AndreasKaratzas",) for rule in rules))
        self.assertTrue(
            all(
                not domain["accepted"]
                and domain["primary"] is None
                and domain["backup"] is None
                for domain in policy["domains"].values()
            )
        )
        self.assertEqual(
            effective_owner(rules, ".github/CODEOWNERS").pattern, "/.github/CODEOWNERS"
        )

    def test_specific_runtime_and_framework_domains_override_broad_paths(self):
        policy = load_policy()
        rules = parse_codeowners(render_codeowners(policy))
        expected = {
            "csrc/runtime/plan.cpp": "api-runtime",
            "tests/frameworks/vllm/operators/normalization/test_rmsnorm.py": "clients",
            "benchmarks/operators/rmsnorm.py": "numerics",
            ".github/workflow-sources/repository/checks.yaml": "delivery",
            ".github/workflow-sources/frameworks/vllm/nightly.yaml": "clients",
            ".github/workflows/frameworks-vllm-nightly.yaml": "clients",
        }
        for path, domain in expected.items():
            with self.subTest(path=path):
                pattern = effective_owner(rules, path).pattern
                self.assertEqual(
                    next(
                        rule["domain"]
                        for rule in reversed(policy["rules"])
                        if rule["pattern"] == pattern
                    ),
                    domain,
                )
