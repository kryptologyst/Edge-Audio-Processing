#!/usr/bin/env python3
"""Quick start script for Edge Audio Processing."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🚀 Edge Audio Processing - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Try: pip install --upgrade pip")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Train a quick model
    if not run_command(
        "python scripts/train.py training.epochs=5 data.n_samples_per_class=20", 
        "Training demo model"
    ):
        print("❌ Training failed")
        sys.exit(1)
    
    # Check if demo can be imported
    if not run_command("python -c 'import demo.app; print(\"Demo app ready\")'", "Checking demo app"):
        print("⚠️  Demo app has issues, but core functionality works")
    
    print("\n" + "=" * 50)
    print("🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Run the interactive demo:")
    print("   streamlit run demo/app.py")
    print("\n2. Train a full model:")
    print("   python scripts/train.py")
    print("\n3. Explore the code in src/ directory")
    print("\n4. Check the README.md for detailed documentation")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
