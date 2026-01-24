# Modifying the Vivado Overlay

Instructions for developers who need to alter the hardware logic or RF parameters of the FPGA design.

## Links

https://www.xilinx.com/support/download.html for downloading Vivado

## Prerequisites

- Vivado Design Suite 2022.1
- RFSoC 4x2 Board Files installed in Vivado

### Simplified Block Design

### Modifying the Block Design

- tcl script to build project from scratch
- TODO: link to guide

### Rename and Upload

Use get_pynq_files script to retrieve necessary files.

- **Bitstream (`.bit`)**: The compiled FPGA configuration
- **Hardware Handoff (`.hwh`)**: Metadata describing the design

use upload script to send it to RFSoC
