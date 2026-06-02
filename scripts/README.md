This folder contains helper scripts that I used during development.

| File                 | Description                                                                   |
| -------------------- | ----------------------------------------------------------------------------- |
| find_ip.bat          | Windows based script to find IP addresses of RFSoC board on the local network |
| find_ip.sh           | Linux based script to find IP addresses of RFSoC board on the local network   |
| get_pynq_files.py    | Script to copy .bit and .hwh files to folder from Vivado project              |
| install.sh           | Installs the firmware package and delivers notebooks with `pynq get-notebooks` |
| install_backend_service.sh | Installs and starts the native systemd backend service                  |
| install_frontend_nginx.sh | Installs the nginx site for the built frontend                         |
| deploy_frontend.sh   | Copies `frontend/dist` to `/var/www/rfsoc-awg`                                |
| prepare_env.sh       | Script to set up environment on RFSoC to run fullstack application            |
| install_tailscale.sh | Script to install Tailscale on RFSoC for remote access                        |
