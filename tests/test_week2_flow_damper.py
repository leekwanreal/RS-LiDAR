import os
import sys
import unittest

# Ensure Diffusion-LiDAR-Sampling is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from tests.test_rs_flow_lidar import TestRSFlowLiDARVelocityGuidance

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING RS-FLOWLIDAR VELOCITY GUIDANCE TEST SUITE")
    print("=" * 60)
    unittest.main(verbosity=2)
