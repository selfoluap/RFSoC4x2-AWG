# RFSoC Dual-DAC Playback Rebuild Package

This folder contains a self-contained Vivado Tcl build flow for recreating the project and generating a bitstream.

## Tested For

- Board: RFSoC4x2 / ZCU208
- Vivado: 2022.1
- Part: `xczu48dr-fsvg1517-2-e`
- Board part: `xilinx.com:zcu208:part0:2.0`

Other Vivado versions may require IP upgrades and may not reproduce the same implementation result.

## Scripts

- `create_project.sh`: Linux launcher for the full build
- `create_project.bat`: Windows launcher for the full build
- `build.tcl`: top-level Vivado batch entrypoint
- `scripts/settings.tcl`: project, device, path, and output settings
- `scripts/create_project.tcl`: creates the Vivado project from source
- `scripts/build_bitstream.tcl`: runs synthesis, implementation, and bitstream generation
- `scripts/export_artifacts.tcl`: exports the bitstream, HWH, and reports

## Build Commands

From this folder on Linux:

```bash
./create_project.sh
```

From this folder on Windows, using a Vivado 2022.1 command prompt:

```bat
create_project.bat
```

Equivalent direct Vivado command:

```bash
vivado -mode batch -source build.tcl -log build.log -journal build.jou
```

## Outputs

The generated Vivado project is written to:

```text
build/vivado/
```

Exported hardware artifacts are written to:

```text
artifacts/pl122p88-ps-s00-100m-dual-dacplay-full-<timestamp>/
```

Expected exported files include `bd_wrapper.bit`, `bd.hwh`, `thesis.bit`, `thesis.hwh`, and final route, timing, and DRC reports.

## Reference Run

This package was fully routed with 0 routing errors, and met timing with `WNS=0.054 ns` and `TNS=0.000 ns`. Howevever, use your newly generated reports as the source of truth for a fresh build.
