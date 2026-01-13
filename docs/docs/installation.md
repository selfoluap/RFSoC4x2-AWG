# Installation Guide

Follow these steps to set up the project.

## Prerequisites

- RFSoC 4x2 Development Board
- MicroSD card (min 16GB) with PYNQ image
- Internet access (via Raspberry Pi bridge if necessary)

![RFSoc](images/rfsoc%20board.png)

## Step 1: Prepare the PYNQ Image

The RFSoC 4x2 must be flashed with a compatible PYNQ SD card image (version 3.0.1 or later).

Follow steps from here: https://www.rfsoc-pynq.io/getting_started.html

## Step 2: Initial Boot and SSH Connection

Connect via SSH:

```bash
ssh xilinx@<ip_address>
```

The IP address is usually shown on the OLED display. If you want to assign a static IP, refer to the following link: https://pynq.readthedocs.io/en/v2.7.0/appendix/assign_a_static_ip.html
The default password is `xilinx`.

## Step 3: System Update

Update the system packages:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

## Step 4: Clone the Repository

```bash
git clone https://github.com/selfoluap/RFSoC4x2-AWG.git
cd RFSoC4x2-AWG
```

## Step 5: Install Application Dependencies

```bash
cd app
pip install -r requirements.txt
```

## Step 6: Test the Installation

Run a quick test to verify the installation:

TODO make a short test script that tests the installation
