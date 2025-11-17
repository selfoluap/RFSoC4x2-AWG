import shutil
import sys
from pathlib import Path

PROJECT_DIR = "/mnt/c/Users/liedt/Documents/Thesis/examples/RFSoC-MTS/boards/RFSoC4x2/build_mts/mts"
DEST_DIR = "/home/liedt/RFSoC4x2-AWG/overlays/"
RUN_NAME = "mts"

def find_vivado_files(project_dir: Path, run_name: str) -> tuple[Path | None, Path | None]:
    bit_file = None
    hwh_file = None

    print(f"Searching in project: {project_dir}")
    print(f"Looking for run directory containing: '{run_name}'")

    run_dirs = list(project_dir.glob(f"*.runs/{run_name}"))
    if not run_dirs:
        print(f"Error: Could not find implementation run directory '{run_name}' in {project_dir / '*.runs'}.", file=sys.stderr)
        run_dirs = list(project_dir.glob("*.runs/impl_*"))
        if not run_dirs:
            print(f"Error: Could not find any implementation run directory in {project_dir / '*.runs'}.", file=sys.stderr)
            return None, None
        else:
            print(f"Warning: Run '{run_name}' not found. Using the first found implementation run: {run_dirs[0].name}", file=sys.stderr)
            run_dir = run_dirs[0]
    elif len(run_dirs) > 1:
        print(f"Warning: Found multiple directories matching '{run_name}'. Using the first one: {run_dirs[0]}", file=sys.stderr)
        run_dir = run_dirs[0]
    else:
        run_dir = run_dirs[0]
        print(f"Found implementation run directory: {run_dir}")

    bit_files = list(run_dir.glob("*.bit"))
    if len(bit_files) == 1:
        bit_file = bit_files[0]
        print(f"Found .bit file: {bit_file}")
    elif len(bit_files) > 1:
        print(f"Error: Found multiple .bit files in {run_dir}. Please clean the directory.", file=sys.stderr)
        return None, None
    else:
        print(f"Error: No .bit file found in {run_dir}", file=sys.stderr)
        return None, None

    hwh_files = list(run_dir.glob("*.hwh"))
    if len(hwh_files) == 1:
        hwh_file = hwh_files[0]
        print(f"Found .hwh file in run directory: {hwh_file}")
    elif len(hwh_files) > 1:
        print(f"Warning: Found multiple .hwh files in {run_dir}. Using the first one: {hwh_files[0]}", file=sys.stderr)
        hwh_file = hwh_files[0]
    else:
        print(f"Searching for .hwh in .gen directories...")
        gen_hwh_files = list(project_dir.glob("*.gen/sources_1/bd/*/hw_handoff/*.hwh"))
        if len(gen_hwh_files) == 1:
            hwh_file = gen_hwh_files[0]
            print(f"Found .hwh file in .gen directory: {hwh_file}")
        elif len(gen_hwh_files) > 1:
            print(f"Warning: Found multiple .hwh files in .gen directories. Using the first one: {gen_hwh_files[0]}", file=sys.stderr)
            hwh_file = gen_hwh_files[0]
        else:
            print(f"Searching for .hwh in .srcs directories...")
            srcs_hwh_files = list(project_dir.glob("*.srcs/sources_1/bd/*/hw_handoff/*.hwh"))
            if len(srcs_hwh_files) == 1:
                hwh_file = srcs_hwh_files[0]
                print(f"Found .hwh file in .srcs directory: {hwh_file}")
            elif len(srcs_hwh_files) > 1:
                print(f"Warning: Found multiple .hwh files in .srcs directories. Using the first one: {srcs_hwh_files[0]}", file=sys.stderr)
                hwh_file = srcs_hwh_files[0]
            else:
                print(f"Error: Could not find .hwh file in standard locations ({run_dir}, *.gen, *.srcs).", file=sys.stderr)
                return bit_file, None

    return bit_file, hwh_file

project_path = Path(PROJECT_DIR).resolve()
print(f"Resolved project path: {project_path}")
dest_path = Path(DEST_DIR).resolve()

if not project_path.is_dir():
    print(f"Error: Project directory not found: {project_path}", file=sys.stderr)
    sys.exit(1)

if not dest_path.exists():
    print(f"Destination directory not found: {dest_path}. Creating it.")
    try:
        dest_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create destination directory: {e}", file=sys.stderr)
        sys.exit(1)
elif not dest_path.is_dir():
    print(f"Error: Destination path exists but is not a directory: {dest_path}", file=sys.stderr)
    sys.exit(1)

bit_file, hwh_file = find_vivado_files(project_path, RUN_NAME)

project_name = project_path.name.lower()
print(f"Using project directory name '{project_name}' for both files.")

copied_files = 0
if bit_file:
    dest_bit_name = f"{project_name}.bit"
    dest_bit_path = dest_path / dest_bit_name
    try:
        shutil.copy2(bit_file, dest_bit_path)
        print(f"Successfully copied {bit_file.name} to {dest_bit_path}")
        copied_files += 1
    except Exception as e:
        print(f"Error copying {bit_file.name}: {e}", file=sys.stderr)

if hwh_file:
    dest_hwh_name = f"{project_name}.hwh"
    dest_hwh_path = dest_path / dest_hwh_name
    try:
        shutil.copy2(hwh_file, dest_hwh_path)
        print(f"Successfully copied {hwh_file.name} to {dest_hwh_path}")
        copied_files += 1
    except Exception as e:
        print(f"Error copying {hwh_file.name}: {e}", file=sys.stderr)

if copied_files == 2:
    print("\nSuccessfully copied both .bit and .hwh files.")
elif copied_files == 1:
    print("\nWarning: Only one file (.bit or .hwh) was found and copied.", file=sys.stderr)
else:
    print("\nError: Failed to find or copy necessary files.", file=sys.stderr)
    sys.exit(1)