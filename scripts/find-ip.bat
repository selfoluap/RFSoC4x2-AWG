@echo off
setlocal enabledelayedexpansion

REM Define the base IP address (e.g. 192.168.1, 192.168.2, 192.168.137)
set BASE_IP=192.168.137.1

REM Set a ping timeout of [ms] before continuing
set TIMEOUT=1

REM Define the start and end range of the IP addresses
set /A START_RANGE=1
set /A END_RANGE=254

REM Loop through the range of IP addresses
for /L %%I in (%START_RANGE%,1,%END_RANGE%) do (
    REM Construct the full IP address
    set IP=%BASE_IP%.%%I

    echo Pinging !IP!...

    REM Wait for the specified timeout
    timeout /t %TIMEOUT% /nobreak >nul

    REM Ping the IP address with a timeout of 100ms (1 second)
    ping -n 1 -w 100 !IP! >nul

    REM Check if the ping was successful
    if !errorlevel!==0 (
        echo Host found at !IP!
    )
)

endlocal