#!/usr/bin/env bash
# Resolve the reviewed release target and emit GitHub job outputs.
set -euo pipefail

should_run=true
skip_reason=""

if [ "${EVENT_NAME}" = "schedule" ] && [ "${SKIP_BIWEEKLY_GATE}" != "true" ]; then
  today="$(date -u +%Y-%m-%d)"
  anchor_seconds="$(date -u -d "${RELEASE_CADENCE_ANCHOR}" +%s)"
  today_seconds="$(date -u -d "${today}" +%s)"
  days_since_anchor=$(( (today_seconds - anchor_seconds) / 86400 ))
  cadence_offset=$(( days_since_anchor % 14 ))

  if [ "${cadence_offset}" -ne 0 ]; then
    should_run=false
    skip_reason="Not on the 14-day release cadence anchored at ${RELEASE_CADENCE_ANCHOR}."
  fi
fi

if [ "${should_run}" != "true" ]; then
  {
    echo "should_run=${should_run}"
    echo "skip_reason=${skip_reason}"
    echo "release_tag="
    echo "release_branch="
    echo "release_version="
    echo "release_kind="
    echo "previous_tag="
    echo "expected_wheel_count=${EXPECTED_WHEEL_COUNT}"
  } >> "${GITHUB_OUTPUT}"

  {
    echo "## Release Target"
    echo "- Should run: ${should_run}"
    echo "- Skip reason: ${skip_reason}"
  } >> "${GITHUB_STEP_SUMMARY}"

  exit 0
fi

git fetch --force --tags origin
git fetch --force origin '+refs/heads/*:refs/remotes/origin/*'

latest_stable_tag="$(git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1 || true)"

release_tag=""
if [ "${EVENT_NAME}" = "push" ]; then
  release_tag="${REF_NAME}"
elif [ -n "${INPUT_RELEASE_TAG}" ]; then
  release_tag="${INPUT_RELEASE_TAG}"
else
  if [ -z "${latest_stable_tag}" ]; then
    echo "Unable to resolve the latest stable release tag."
    exit 1
  fi
  version="${latest_stable_tag#v}"
  IFS=. read -r major minor patch <<< "${version}"
  release_tag="v${major}.${minor}.$((patch + 1))"
fi

if [[ ! "${release_tag}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$ ]]; then
  echo "Invalid release tag: ${release_tag}"
  echo "Expected vX.Y.Z or vX.Y.Z.postN."
  exit 1
fi

release_kind=scheduled
if [[ "${release_tag}" =~ \.post[0-9]+$ ]]; then
  release_kind=post
fi

base_tag="${release_tag%%.post*}"
if [ -n "${INPUT_RELEASE_BRANCH}" ]; then
  release_branch="${INPUT_RELEASE_BRANCH}"
else
  release_branch="release/${base_tag}"
fi

branch_exists=false
if git show-ref --verify --quiet "refs/remotes/origin/${release_branch}"; then
  branch_exists=true
fi

tag_exists=false
if git rev-parse -q --verify "refs/tags/${release_tag}^{commit}" >/dev/null; then
  tag_exists=true
fi

if [ "${release_kind}" = "post" ] && { [ "${branch_exists}" != "true" ] || [ "${tag_exists}" != "true" ]; }; then
  echo "Post release ${release_tag} must already exist on ${release_branch}; refusing to create it automatically."
  exit 1
fi

if [ "${release_kind}" = "scheduled" ] && [ "${EVENT_NAME}" != "push" ] && [ "${CREATE_MISSING_REFS}" = "true" ]; then
  target_commit=""
  if git rev-parse -q --verify "refs/remotes/origin/${INPUT_RELEASE_REF}^{commit}" >/dev/null; then
    target_commit="$(git rev-parse "refs/remotes/origin/${INPUT_RELEASE_REF}^{commit}")"
  else
    target_commit="$(git rev-parse "${INPUT_RELEASE_REF}^{commit}")"
  fi

  if [ "${branch_exists}" != "true" ]; then
    git branch "${release_branch}" "${target_commit}"
    git push origin "refs/heads/${release_branch}"
    git fetch --force origin "refs/heads/${release_branch}:refs/remotes/origin/${release_branch}"
    branch_exists=true
  fi

  if [ "${tag_exists}" != "true" ]; then
    branch_commit="$(git rev-parse "refs/remotes/origin/${release_branch}^{commit}")"
    git tag "${release_tag}" "${branch_commit}"
    git push origin "refs/tags/${release_tag}"
    tag_exists=true
  fi
fi

if [ "${branch_exists}" != "true" ]; then
  echo "Missing release branch: ${release_branch}"
  exit 1
fi
if [ "${tag_exists}" != "true" ]; then
  echo "Missing release tag: ${release_tag}"
  exit 1
fi

tag_commit="$(git rev-parse "${release_tag}^{commit}")"
branch_commit="$(git rev-parse "refs/remotes/origin/${release_branch}^{commit}")"

if [ "${tag_commit}" != "${branch_commit}" ]; then
  echo "Tag ${release_tag} points to ${tag_commit}, but ${release_branch} points to ${branch_commit}."
  echo "Release tags must point at the release branch HEAD."
  exit 1
fi

previous_tag=""
if [ "${release_kind}" = "scheduled" ]; then
  for tag in $(git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V); do
    if [ "${tag}" = "${release_tag}" ]; then
      break
    fi
    previous_tag="${tag}"
  done
else
  previous_tag="${base_tag}"
  for tag in $(git tag -l "${base_tag}.post*" | sort -V); do
    if [ "${tag}" = "${release_tag}" ]; then
      break
    fi
    previous_tag="${tag}"
  done
fi

if [ -z "${previous_tag}" ]; then
  echo "Unable to resolve previous tag for ${release_tag}."
  exit 1
fi

if [ "${EVENT_NAME}" != "push" ] && [ "${FORCE_REBUILD}" != "true" ]; then
  published_wheels="$(gh release view "${release_tag}" --json assets --jq '[.assets[].name | select(endswith(".whl"))] | length' 2>/dev/null || echo 0)"
  is_draft=$(gh release view "${release_tag}" --json isDraft --jq .isDraft 2>/dev/null || echo true)
  if [ "${is_draft}" = false ] && [ "${published_wheels}" -ge "${EXPECTED_WHEEL_COUNT}" ]; then
    should_run=false
    skip_reason="GitHub Release ${release_tag} already has ${published_wheels} wheel assets."
  fi
fi

{
  echo "should_run=${should_run}"
  echo "skip_reason=${skip_reason}"
  echo "source_revision=${tag_commit}"
  echo "release_tag=${release_tag}"
  echo "release_branch=${release_branch}"
  echo "release_version=${release_tag#v}"
  echo "release_kind=${release_kind}"
  echo "previous_tag=${previous_tag}"
  echo "expected_wheel_count=${EXPECTED_WHEEL_COUNT}"
} >> "${GITHUB_OUTPUT}"

{
  echo "## Release Target"
  echo "- Tag: ${release_tag}"
  echo "- Branch: ${release_branch}"
  echo "- Commit: ${tag_commit}"
  echo "- Previous tag: ${previous_tag}"
  echo "- Should run: ${should_run}"
  if [ -n "${skip_reason}" ]; then
    echo "- Skip reason: ${skip_reason}"
  fi
} >> "${GITHUB_STEP_SUMMARY}"
