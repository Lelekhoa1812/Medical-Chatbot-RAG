#!/usr/bin/env python3
"""
Test script to verify CSV file can be loaded in Docker environment
"""
import os
import sys
import pandas as pd

def test_csv_in_docker():
    """Test CSV loading in Docker environment"""
    try:
        # Get the utils directory path
        utils_dir = "/app/utils"
        csv_path = os.path.join(utils_dir, "symbipredict_2022.csv")
        
        print(f"Testing CSV loading in Docker...")
        print(f"Utils directory: {utils_dir}")
        print(f"CSV path: {csv_path}")
        print(f"CSV file exists: {os.path.exists(csv_path)}")
        
        if os.path.exists(csv_path):
            # Try to read the CSV
            df = pd.read_csv(csv_path)
            print(f"✅ CSV file loaded successfully in Docker!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)[:5]}...")
            return True
        else:
            print(f"❌ CSV file not found at: {csv_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading CSV in Docker: {e}")
        return False

if __name__ == "__main__":
    success = test_csv_in_docker()
    sys.exit(0 if success else 1)
