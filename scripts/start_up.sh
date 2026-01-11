#!/bin/bash
su root
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
streamlit run /home/xilinx/paulo/RFSoC4x2-AWG/app/app.py