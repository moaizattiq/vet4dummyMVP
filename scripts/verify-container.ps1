# Build and smoke-test the production container from the repository root.
$ErrorActionPreference = "Stop"

$imageName = "app-test"
$containerName = "test-container"
$containerStarted = $false

if (-not (Test-Path -LiteralPath ".env")) {
    throw ".env is required. Copy .env.example to .env and add V4W_DATABASE_URL."
}

try {
    docker build -t $imageName .
    if ($LASTEXITCODE -ne 0) {
        throw "docker build failed with exit code $LASTEXITCODE"
    }

    docker run -d --env-file .env -p 8000:8000 --name $containerName $imageName
    if ($LASTEXITCODE -ne 0) {
        throw "docker run failed with exit code $LASTEXITCODE"
    }
    $containerStarted = $true

    # Show startup output before probing the HTTP surface.
    Start-Sleep -Seconds 2
    docker logs $containerName
    if ($LASTEXITCODE -ne 0) {
        throw "docker logs failed with exit code $LASTEXITCODE"
    }

    # Retry connection failures briefly while Uvicorn finishes starting.
    curl.exe --fail --show-error --silent --head `
        --retry 10 --retry-all-errors --retry-delay 1 `
        http://localhost:8000
    if ($LASTEXITCODE -ne 0) {
        throw "HTTP verification failed with exit code $LASTEXITCODE"
    }

    Write-Host "Container verification passed: http://localhost:8000 returned HTTP 200."
}
finally {
    if ($containerStarted) {
        docker rm -f $containerName
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Could not remove $containerName; remove it manually with: docker rm -f $containerName"
        }
    }
}
