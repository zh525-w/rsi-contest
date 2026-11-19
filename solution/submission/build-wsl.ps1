# 在本机 WSL2 中构建镜像（不需要 Docker Desktop）。
# 前置：wsl -d Ubuntu 可用，且已安装 docker.io 并启动 dockerd。
# 用法：powershell -ExecutionPolicy Bypass -File build-wsl.ps1
param([string]$Distro = "Ubuntu")

$ErrorActionPreference = "Stop"
$sub = $PSScriptRoot
$wslPath = (wsl -d $Distro -e sh -lc "wslpath -a '$($sub -replace '\\','/')'" 2>&1 | Out-String).Trim()

$script = @"
set -eu
systemctl start docker 2>/dev/null || true
for i in 1 2 3 4 5; do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { echo "docker daemon not running"; exit 1; }
SRC=$wslPath
DST=/root/build/contest-harness
rm -rf "`$DST"; mkdir -p "`$DST"; cp -r "`$SRC"/. "`$DST"/
cd "`$DST"
docker build --platform linux/amd64 -t contest-harness:1.0 -f Dockerfile .
docker run --rm --platform linux/amd64 contest-harness:1.0 describe
docker image inspect --format '{{.Os}}/{{.Architecture}} {{.Size}}' contest-harness:1.0
docker save -o "`$SRC/contest-harness-1.0.tar" contest-harness:1.0
ls -lh "`$SRC/contest-harness-1.0.tar"
"@
$tmp = Join-Path $env:TEMP "build-harness-wsl.sh"
[IO.File]::WriteAllText($tmp, $script.Replace("`r`n", "`n"))
$tmpWsl = (wsl -d $Distro -e sh -lc "wslpath -a '$($tmp -replace '\\','/')'" 2>&1 | Out-String).Trim()
wsl -d $Distro -u root -- sh $tmpWsl
