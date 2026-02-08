#!/usr/bin/env python3
"""
Direct runner script - no installation needed.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_test.main import main

if __name__ == "__main__":
    main()
