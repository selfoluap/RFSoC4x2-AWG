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


if __name__ == "__main__":
    unittest.main()
