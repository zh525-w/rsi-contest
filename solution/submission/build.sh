#!/usr/bin/env sh
# 构建并验证 Harness 镜像；--save 时导出 tar 供平台上传。
# 用法：sh build.sh [--save]
set -eu
cd "$(dirname "$0")"

TAG="contest-harness:1.0"
TAR="contest-harness-1.0.tar"

echo "[1/4] build $TAG (linux/amd64)"
docker build --platform linux/amd64 -t "$TAG" -f Dockerfile .

echo "[2/4] image architecture"
docker image inspect --format '{{.Os}}/{{.Architecture}}' "$TAG"

echo "[3/4] describe self-check inside the image"
docker run --rm --platform linux/amd64 "$TAG" describe

if [ "${1:-}" = "--save" ]; then
  echo "[4/4] docker save -> $TAR"
  docker save -o "$TAR" "$TAG"
  ls -lh "$TAR"
else
  echo "[4/4] skip export (add --save to write the tar)"
fi
