# Define the path to the .env file
$envPath = ".\.env"

# Check if the .env file exists
if (Test-Path $envPath) {
    # Read the .env file
    $envContent = Get-Content $envPath

    # Parse the environment variables
    foreach ($line in $envContent) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }

    # Run docker-compose
    docker-compose up --build
}
else {
    Write-Error "The .env file was not found at $envPath"
}