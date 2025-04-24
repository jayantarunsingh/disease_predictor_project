@echo off
setlocal EnableDelayedExpansion

REM Variables
set STACK_NAME=my_stack
set COMPOSE_FILE=docker-compose.yml
set NETWORK_NAME=disease-net

REM Function to clean up existing stack and swarm
echo Cleaning up existing stack and swarm...

REM Check and remove existing stack
for /f "delims=" %%i in ('docker stack ls ^| findstr /C:"%STACK_NAME%"') do (
    echo Removing existing stack...
    docker stack rm %STACK_NAME%
    timeout /t 10 >nul
)

REM Check if in swarm mode
for /f "delims=" %%i in ('docker info ^| findstr /C:"Swarm: active"') do (
    echo Leaving existing swarm...
    docker swarm leave --force
    timeout /t 5 >nul
)

REM Main deployment process
echo Starting deployment process...

REM Initialize swarm
echo Initializing new swarm...
docker swarm init

REM Create overlay network if it doesn't exist
echo Creating overlay network...
docker network create --driver overlay %NETWORK_NAME% 2>nul || echo Network already exists

REM Deploy stack
echo Deploying stack "%STACK_NAME%"...
docker stack deploy -c "%COMPOSE_FILE%" "%STACK_NAME%"

REM Wait for services to be created
timeout /t 5 >nul

REM Check service status
echo Checking service status...
docker service ls

echo Deployment completed successfully!