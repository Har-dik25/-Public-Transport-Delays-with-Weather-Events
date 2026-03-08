import pandas as pd
import joblib
import os
import glob
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

print("="*60)
print("🚀 PRODUCTION OPTIMIZATION SCRIPT")
print("="*60)

# 1. Convert CSV Data to Parquet
print("\n📦 1. Converting Datasets from CSV to Parquet...")
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

for csv_file in csv_files:
    start_time = time.time()
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        parquet_file = csv_file.replace(".csv", ".parquet")
        
        # Save as parquet (snappy compression is default and fast)
        df.to_parquet(parquet_file, engine='pyarrow', index=False)
        
        csv_size = os.path.getsize(csv_file) / (1024 * 1024)
        pq_size = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"   ✅ Converted {os.path.basename(csv_file)}")
        print(f"      Size reduction: {csv_size:.1f}MB -> {pq_size:.1f}MB ({pq_size/csv_size*100:.1f}%) | Time: {time.time()-start_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Failed to convert {os.path.basename(csv_file)}: {e}")

# 2. Compress Models
print("\n🗜️ 2. Compressing ML Models...")
pkl_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
pkl_files += glob.glob(os.path.join(DATA_DIR, "shap_*.pkl")) # compress SHAP values too

for pkl_file in pkl_files:
    start_time = time.time()
    try:
        # Load the uncompressed model
        model = joblib.load(pkl_file)
        
        # We'll save it with the exact same name, but compressed.
        # compress=3 offers a great balance of size reduction vs load speed
        joblib.dump(model, pkl_file, compress=3)
        
        new_size = os.path.getsize(pkl_file) / (1024 * 1024)
        print(f"   ✅ Compressed {os.path.basename(pkl_file)}")
        print(f"      Final Size: {new_size:.1f}MB | Time: {time.time()-start_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Failed to compress {os.path.basename(pkl_file)}: {e}")

print("\n🎉 Optimization Complete! Now update app.py to read the .parquet files.")
