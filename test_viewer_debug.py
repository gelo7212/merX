#!/usr/bin/env python3
"""Debug script to test database viewer setup."""

import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from examples.database_viewer import EnhancedDatabaseViewer

logging.basicConfig(level=logging.INFO)

def test_viewer_setup():
    print("Testing database viewer setup...")
    
    viewer = EnhancedDatabaseViewer()
    print(f"Viewer created: {viewer is not None}")
    print(f"Initial engine: {viewer.engine}")
    
    try:
        print("Attempting setup...")
        viewer.setup_viewer('data/test_output')
        print(f"Engine after setup: {viewer.engine is not None}")
        if viewer.engine:
            print(f"Engine type: {type(viewer.engine)}")
            print(f"Storage type: {type(viewer.engine.storage)}")
            print(f"Storage has ramx: {hasattr(viewer.engine.storage, 'ramx')}")
        else:
            print("Engine is None!")
    except Exception as e:
        print(f"Setup failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_viewer_setup()
