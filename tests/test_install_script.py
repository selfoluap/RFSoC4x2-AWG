from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "scripts" / "install.sh"


class InstallScriptTest(unittest.TestCase):
    def test_install_defaults_to_non_editable(self):
        script = INSTALL_SH.read_text()

        self.assertIn('RFSOC_AWG_EDITABLE:-0', script)
        self.assertIn('python3 -m pip install "${PIP_INSTALL_ARGS[@]}"', script)
        self.assertIn('PIP_INSTALL_ARGS=(-e "$REPO_ROOT")', script)
        self.assertNotIn('RFSOC_AWG_WHEEL', script)

    def test_pip_install_is_non_fatal(self):
        script = INSTALL_SH.read_text()

        # pip install must not abort the script on failure — the .pth file
        # and runtime copy are the real import mechanisms.
        self.assertIn('--no-build-isolation', script)
        self.assertIn('Relying on .pth file', script)
        # The overlays copy must be non-fatal (|| true) so a missing
        # site-packages firmware dir doesn't abort the install.
        self.assertIn('FIRMWARE_DIR', script)
        self.assertIn('2>/dev/null || true', script)

    def test_setup_py_shim_exists_for_old_setuptools(self):
        # PYNQ images with setuptools < 61 cannot read PEP 621 [project]
        # metadata from pyproject.toml. The setup.py shim provides explicit
        # metadata so those images don't install a wheel named "UNKNOWN".
        setup_py = (ROOT / "setup.py").read_text()
        self.assertIn("rfsoc4x2-awg", setup_py)
        self.assertIn("find_packages", setup_py)
        self.assertIn("firmware*", setup_py)

    def test_refuses_root(self):
        script = INSTALL_SH.read_text()

        # Running as root sends all artifacts to /root where the xilinx
        # Jupyter server cannot find them.
        self.assertIn('EUID', script)
        self.assertIn('do not run this script as root', script)
        self.assertIn('su - xilinx', script)


if __name__ == "__main__":
    unittest.main()
