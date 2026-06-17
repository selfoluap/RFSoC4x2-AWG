from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "scripts" / "install.sh"


class InstallScriptTest(unittest.TestCase):
    def test_install_defaults_to_non_editable(self):
        script = INSTALL_SH.read_text()

        self.assertIn('RFSOC_AWG_EDITABLE:-0', script)
        self.assertIn('python3 -m pip install "$REPO_ROOT"', script)
        self.assertIn('python3 -m pip install -e "$REPO_ROOT"', script)
        self.assertNotIn('RFSOC_AWG_WHEEL', script)


if __name__ == "__main__":
    unittest.main()
