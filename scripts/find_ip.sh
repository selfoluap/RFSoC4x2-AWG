#!/usr/bin/env bash

# Define the base IP address (e.g., 192.168.137)
BASE_IP="192.168.137"

# Set a ping timeout in milliseconds
TIMEOUT=100

# Define the start and end range of the IP addresses
START_RANGE=1
END_RANGE=254

# Loop through the range of IP addresses
for ((i=START_RANGE; i<=END_RANGE; i++)); do
    # Construct the full IP address
    IP="${BASE_IP}.${i}"

    echo "Pinging ${IP}..."

    # Ping the IP address once with timeout
    if ping -c 1 -W 1 "$IP" >/dev/null 2>&1; then
        echo "Host found at ${IP}"
    fi
done
