# 构建并验证 Harness 镜像（需要 Docker Desktop / Linux docker）。
# 用法：
#   powershell -ExecutionPolicy Bypass -File build.ps1            # 只构建+自检
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Save      # 额外导出 tar
#
# 平台要求：单个 Linux/amd64 镜像的 `docker save` 未压缩 tar。
# 不要上传 zip、源码 tar 或 `docker export` 产物。

param(
    [switch]$Save,
    [string]$Tag = "contest-harness:1.0",
    [string]$Tar = "contest-harness-1.0.tar"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    Write-Host "[1/4] build $Tag (linux/amd64)"
    docker build --platform linux/amd64 -t $Tag -f Dockerfile .

    Write-Host "[2/4] image architecture"
    docker image inspect --format "{{.Os}}/{{.Architecture}}" $Tag

    Write-Host "[3/4] describe self-check inside the image"
    docker run --rm --platform linux/amd64 $Tag describe

    if ($Save) {
        Write-Host "[4/4] docker save -> $Tar"
        docker save -o $Tar $Tag
        Get-Item $Tar | Select-Object Name, @{n = "MiB"; e = { [math]::Round($_.Length / 1MB, 1) } }
    }
    else {
        Write-Host "[4/4] skip export (add -Save to write the tar)"
    }
}
finally {
    Pop-Location
}
