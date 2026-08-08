import sys
import unittest
from unittest.mock import patch

import coppelia_env
from coppelia_env.track_env import _remote_api_client_class


class CoppeliaOptionalDependencyTests(unittest.TestCase):
    def test_top_level_package_exposes_core_interfaces(self):
        self.assertIsNotNone(coppelia_env.CoppeliaTrackEnv)
        self.assertIsNotNone(coppelia_env.NormalizedCmdVelAdapter)

    def test_connect_loader_reports_missing_remote_api_client(self):
        with patch.dict(sys.modules, {"coppeliasim_zmqremoteapi_client": None}):
            with self.assertRaisesRegex(ModuleNotFoundError, "optional.*coppeliasim_zmqremoteapi_client"):
                _remote_api_client_class()


if __name__ == "__main__":
    unittest.main()
