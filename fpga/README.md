# RFSoC Dual-DAC Playback Rebuild Package

This folder contains a self-contained Vivado Tcl build flow for recreating the project and generating a bitstream.

## Tested For

- Board: RFSoC4x2 / ZCU208
- Vivado: 2022.1
- Part: `xczu48dr-fsvg1517-2-e`
- Board part: `xilinx.com:zcu208:part0:2.0`

Other Vivado versions may require IP upgrades and may not reproduce the same implementation result.

## Scripts

- `create_project.sh`: Linux launcher for project creation
- `create_project.bat`: Windows launcher for project creation
- `build_bitstream.sh`: Linux launcher for synthesis, implementation, bitstream generation, and artifact export
- `build_bitstream.bat`: Windows launcher for synthesis, implementation, bitstream generation, and artifact export
- `build_all.sh`: Linux launcher for the complete create-and-build flow
- `build_all.bat`: Windows launcher for the complete create-and-build flow
- `clean.sh`: Linux cleanup for generated Vivado files
- `clean.bat`: Windows cleanup for generated Vivado files
- `create_project.tcl`: top-level Vivado project creation entrypoint
- `build_bitstream.tcl`: top-level Vivado bitstream build entrypoint
- `build_all.tcl`: top-level Vivado complete rebuild entrypoint
- `scripts/settings.tcl`: project, device, path, and output settings
- `scripts/create_project_impl.tcl`: creates the Vivado project from source
- `scripts/build_bitstream_impl.tcl`: runs synthesis, implementation, and bitstream generation
- `scripts/export_artifacts.tcl`: exports the bitstream, HWH, and reports

## Build Commands

From this folder on Linux, create the Vivado project first:

```bash
./create_project.sh
```

Then build the bitstream:

```bash
./build_bitstream.sh
```

On Windows, using a Vivado 2022.1 command prompt:

```bat
create_project.bat
build_bitstream.bat
```

Equivalent direct Vivado commands:

```bash
vivado -mode batch -source create_project.tcl -log create_project.log -journal create_project.jou
vivado -mode batch -source build_bitstream.tcl -log build_bitstream.log -journal build_bitstream.jou
```

To run both steps in one command:

```bash
./build_all.sh
```

To remove generated project files and Vivado logs from this folder:

```bash
./clean.sh
```

On Windows:

```bat
clean.bat
```

## Outputs

The generated Vivado project is written to:

```text
build/vivado/rfsocawg.xpr
```

Exported hardware artifacts are written to:

```text
artifacts/rfsocawg-dual-dacplay-full-<timestamp>/
```

Expected exported files include `bd_wrapper.bit`, `bd.hwh`, `rfsocawg.bit`, `rfsocawg.hwh`, and final route, timing, and DRC reports.

## Reference Run

The completed local reference run was fully routed with 0 routing errors, and met timing with `WNS=0.054 ns` and `TNS=0.000 ns`.

Use your newly generated reports as the source of truth for a fresh build.
