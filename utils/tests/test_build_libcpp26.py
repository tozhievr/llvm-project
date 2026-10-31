import tempfile
import unittest
from pathlib import Path

from utils.build_libcpp26 import (
    configuration_module,
    discover_modules,
    topological_order,
)


class BuildLibCpp26Tests(unittest.TestCase):
    def test_build_configuration_matches_avx2_target(self):
        source = configuration_module()

        self.assertIn("kIsWindowsPlatform = true", source)
        self.assertIn("kIsEnabledFeature<AVX2> = true", source)
        self.assertIn("kIsEnabledFeature<AVX512> = false", source)

    def test_discovers_dependencies_before_importers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.cppm").write_text(
                "export module test.base;\n", encoding="utf-8"
            )
            (root / "consumer.cppm").write_text(
                "export module test.consumer;\nimport test.base;\n",
                encoding="utf-8",
            )

            ordered = topological_order(discover_modules(root))
            names = [module.name for module in ordered]

            self.assertLess(names.index("cmake_generated"), names.index("test.base"))
            self.assertLess(names.index("test.base"), names.index("test.consumer"))

    def test_reports_import_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.cppm").write_text(
                "export module test.a;\nimport test.b;\n", encoding="utf-8"
            )
            (root / "b.cppm").write_text(
                "export module test.b;\nimport test.a;\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "module import cycle"):
                topological_order(discover_modules(root))


if __name__ == "__main__":
    unittest.main()
