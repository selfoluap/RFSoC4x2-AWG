# RFSoC4x2 Based AWG

A simple 9.8304 GSPS AWG implementation using the RFSoC 4x2 development board and the PYNQ framework.

## Overview

- **Waveform generation**: 9.8304 GSPS DAC output with 14-bit resolution
- **User-friendly interfaces**: Jupyter notebooks with example code
- **Web application**: React frontend with FastAPI backend

## Documentation

**[Documentation](https://rfsoc4x2-awg-docs.readthedocs.io/en/latest/)**

## Quick Start

Follow the [installation guide](https://rfsoc4x2-awg-docs.readthedocs.io/en/latest/installation/) to get started.

## Repository Structure

```
RFSoC4x2-AWG/
├── backend/               # FastAPI backend and waveform
├── hdl/                   # Hardware description language files
├── notebooks/             # Jupyter notebooks with example code
├── frontend/              # React frontend
├── firmware/              # Firmware-facing hardware helper modules
├── overlays/              # FPGA bitstreams and hardware handoffs
├── scripts/               # Utility and deployment scripts
└── tests/                 # Test files
```

## Contributing

If you find bugs or have any other ideas in mind what to add, please feel free to open a pull request.

## License

MIT
