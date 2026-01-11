"""
Risk-Conditioned AI Evaluation Lab - Main Entry Point

A System for Measuring, Analyzing, and Governing Manipulative Behavior in AI Systems
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from risklab.cli import main as cli_main


def main():
    """Main entry point."""
    cli_main()


if __name__ == "__main__":
    main()
