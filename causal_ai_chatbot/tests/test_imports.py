#!/usr/bin/env python3
"""
Test script to verify imports work correctly
Run this before starting the chatbot server
"""

import sys
from pathlib import Path

print("=" * 60)
print("Testing TRAM-DAG and R Integration Imports")
print("=" * 60)
print()

# Test TRAM-DAG import
print("1. Testing TRAM-DAG import...")
print(f"   Current file: {__file__}")
print(f"   Current directory: {Path(__file__).parent}")

tramdag_parent = Path(__file__).parent.parent.parent  # Repository root (contains tramdag/)
tramdag_path = tramdag_parent / "tramdag"

print(f"   Calculated TramDag parent: {tramdag_parent}")
print(f"   TramDag path: {tramdag_path}")
print(f"   TramDag exists: {tramdag_path.exists()}")

if tramdag_path.exists():
    print(f"   TramDag __init__.py exists: {(tramdag_path / '__init__.py').exists()}")
    
    # Add to path
    if str(tramdag_parent) not in sys.path:
        sys.path.insert(0, str(tramdag_parent))
        print(f"   Added {tramdag_parent} to sys.path")
    else:
        print(f"   {tramdag_parent} already in sys.path")
    
    # Try import
    try:
        from tramdag import TramDagConfig, TramDagModel, TramDagDataset
        print("   [OK] TRAM-DAG import successful!")
        print(f"   TramDagConfig: {TramDagConfig}")
        print(f"   TramDagModel: {TramDagModel}")
    except ImportError as e:
        print(f"   [ERROR] TRAM-DAG import failed: {e}")
        print(f"   sys.path entries containing 'tramdag':")
        for p in sys.path:
            if 'tramdag' in p.lower():
                print(f"     - {p}")
else:
    print(f"   [ERROR] TramDag directory not found at {tramdag_path}")

print()

# Test R integration import
print("2. Testing R Integration import...")
application_dir = Path(__file__).parent.parent.parent / "application" / "r_integration"
print(f"   Calculated application dir: {application_dir}")
print(f"   Application dir exists: {application_dir.exists()}")

if application_dir.exists():
    r_bridge_path = application_dir / "r_python_bridge.py"
    print(f"   r_python_bridge.py exists: {r_bridge_path.exists()}")
    
    if r_bridge_path.exists():
        if str(application_dir) not in sys.path:
            sys.path.insert(0, str(application_dir))
            print(f"   Added {application_dir} to sys.path")
        
        try:
            from r_python_bridge import RConsistencyChecker
            print("   [OK] R Integration import successful!")
            print(f"   RConsistencyChecker: {RConsistencyChecker}")
        except ImportError as e:
            print(f"   [ERROR] R Integration import failed: {e}")
else:
    print(f"   [WARN] Application directory not found (this is OK if R integration is optional)")

print()
print("=" * 60)
print("Import Test Complete")
print("=" * 60)
