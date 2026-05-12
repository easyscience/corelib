#!/usr/bin/env bash

set -euo pipefail

remote_repository="${DASHBOARD_REMOTE_REPOSITORY:?}"
publish_branch="${DASHBOARD_PUBLISH_BRANCH:-master}"
source_dir="${DASHBOARD_SOURCE_DIR:?}"
token="${DASHBOARD_TOKEN:?}"
git_user_name="${DASHBOARD_GIT_USER_NAME:-easyscience[bot]}"
git_user_email="${DASHBOARD_GIT_USER_EMAIL:?}"
commit_message="${DASHBOARD_COMMIT_MESSAGE:?}"
max_attempts="${DASHBOARD_PUSH_ATTEMPTS:-3}"
delay_seconds="${DASHBOARD_PUSH_DELAY_SECONDS:-15}"

workspace_dir="$(mktemp -d)"
repo_dir="${workspace_dir}/dashboard"
remote_url="https://x-access-token:${token}@github.com/${remote_repository}.git"

cleanup() {
  rm -rf "${workspace_dir}"
}

prepare_worktree() {
  if [[ ! -d "${repo_dir}/.git" ]]; then
    git clone --branch "${publish_branch}" --depth 1 "${remote_url}" "${repo_dir}"
  else
    git -C "${repo_dir}" fetch origin "${publish_branch}"
    git -C "${repo_dir}" checkout "${publish_branch}"
    git -C "${repo_dir}" reset --hard "origin/${publish_branch}"
    git -C "${repo_dir}" clean -fd
  fi

  git -C "${repo_dir}" config user.name "${git_user_name}"
  git -C "${repo_dir}" config user.email "${git_user_email}"
}

sync_publish_dir() {
  cp -R "${source_dir}/." "${repo_dir}/"
  git -C "${repo_dir}" add .

  if git -C "${repo_dir}" diff --cached --quiet; then
    return 1
  fi

  git -C "${repo_dir}" commit -m "${commit_message}"
}

trap cleanup EXIT

prepare_worktree

if ! sync_publish_dir; then
  echo "No dashboard changes to publish."
  exit 0
fi

for ((attempt = 1; attempt <= max_attempts; attempt += 1)); do
  if git -C "${repo_dir}" push origin "HEAD:${publish_branch}"; then
    echo "Dashboard published on attempt ${attempt}."
    exit 0
  fi

  if ((attempt == max_attempts)); then
    echo "Dashboard publish failed after ${max_attempts} attempts." >&2
    exit 1
  fi

  echo "Dashboard push attempt ${attempt} failed. Retrying in ${delay_seconds}s." >&2
  sleep "${delay_seconds}"

  prepare_worktree

  if ! sync_publish_dir; then
    echo "Dashboard changes already exist in the target repository."
    exit 0
  fi
done