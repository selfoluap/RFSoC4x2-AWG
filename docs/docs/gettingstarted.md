# Quickstart

This section provides the minimal steps required to bring the RFSoC-based serrodyne modulation system into a working state. Following these instructions, a user should be able to generate and output a basic serrodyne waveform within a few minutes.

The Quickstart assumes that the hardware has already been physically connected as described in _Hardware Setup & System Architecture_, and that a PYNQ image with the required overlay is installed on the RFSoC.

---

## Prerequisites

Before starting, ensure that the following conditions are met:

- The RFSoC board is powered on and connected via Ethernet
- The board is reachable over the network (via IP address or hostname)

---

## Installation

---

## Connecting to the RFSoC

- follow other tutorial to get ssh access to board
  Later on, if Tailscale is enabled, the hostname assigned within the tailnet can be used directly. Otherwise, use the fixed IP address assigned in the local laboratory network.

---

## Clone Repo and Start Service

- write out that start up script needs to be run which then makes sure that the app is from now on being run as a systemd service

## Starting the Backend Service

The control logic is exposed via a FastAPI backend running on the RFSoC’s ARM processor.

Once started, the backend listens on a predefined local port and provides an HTTP API for waveform configuration and playback.

To verify that the backend is running, open a browser and navigate to:

http://&lt;rfsoc-hostname&gt;:&lt;backend-port&gt;/docs

This page shows the automatically generated FastAPI documentation and lists all available API endpoints.

The graphical user interface is implemented as a Streamlit application and runs on the RFSoC. Start the GUI from the same application directory:

streamlit run app.py

After startup, Streamlit prints a local and network URL. Open the network URL in a web browser on any machine that can reach the RFSoC:

http://&lt;rfsoc-hostname&gt;:&lt;streamlit-port&gt;

The GUI provides controls for configuring waveform parameters and triggering playback.

---

## Generating a Test Waveform

To verify correct operation, generate a simple single-frequency serrodyne waveform using the GUI:

1. Select a single ramp segment
2. Set the desired frequency shift
3. Choose a moderate amplitude
4. Apply the configuration and start playback

If an RF amplifier and EOM are connected, a frequency-shifted optical signal should be observable using a Fabry–Pérot interferometer or equivalent diagnostic.  
For software-only testing, successful playback can be verified by observing backend logs or by using the DAC–ADC loopback configuration described in the Evaluation chapter of the thesis.
