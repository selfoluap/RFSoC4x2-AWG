# Remote Access with Tailscale

Tailscale is used to provide remote access to the RFSoC. It creates a mesh VPN based on the WireGuard protocol called "Tailnet" - where the RFSoC is assigned a stable hostname and IP address, regardless of its location in the local laboratory infrastructure.

## Installation

```bash
./scripts/install_tailscale.sh
```

### Authenticate the Board

Start Tailscale and authenticate:

```bash
sudo tailscale up --hostname rfsoc-awg
```

## Accessing Services

Once connected to your Tailnet, access the RFSoC services using the Tailscale hostname:

| Service             | URL                     |
| ------------------- | ----------------------- |
| Streamlit Dashboard | `http://rfsoc-awg:8501` |
| FastAPI Backend     | `http://rfsoc-awg:8000` |
| Jupyter Notebook    | `http://rfsoc-awg:8888` |
| SSH                 | `ssh xilinx@rfsoc-awg`  |
