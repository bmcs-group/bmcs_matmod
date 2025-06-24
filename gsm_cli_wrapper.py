#!/usr/bin/env python3
"""
Development wrapper for GSM CLI
"""
import sys
import os

# Add the development directory to Python path
sys.path.insert(0, "/home/rch/Coding/bmcs_matmod")

# Import and run the CLI
from bmcs_matmod.gsm_lagrange.cli_gsm import main

if __name__ == "__main__":
    main()
