# Validate generated workflows

This action runs `python3 -S -m ci.workflows --check` from `GITHUB_WORKSPACE`. Check out the reviewed AITER repository at that path first and provide Python 3. It checks source/entrypoint parity without writing generated files, installing dependencies or downloading actionlint.

```yaml
- name: Check generated workflows
  uses: ./.github/actions/common/validate-workflows
```

The action explicitly changes to the checkout root, so a caller's default shell working directory cannot redirect the check. A missing workspace or missing `ci/workflows/__main__.py` fails with a clear message. The command's failure status is preserved. There are no inputs, outputs, credentials or alternate candidate directories.
