#!/usr/bin/env python3
"""
Command Line Interface for Research Paper Q&A Agent

Entry point for the application providing a clean CLI interface.
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    
    from main import main
    sys.exit(main())
