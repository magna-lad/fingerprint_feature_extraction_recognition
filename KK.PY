import pickle
from pprint import pprint

file_path = r"C:\Users\kound\Downloads\archive (1)\biometric_cache\processed_data.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

# Print actual stored data
if isinstance(data, dict):
    for i, (key, value) in enumerate(data.items()):
        print(f"\n{'='*60}")
        print(f"RECORD {i+1}")
        print("KEY:", key)
        print("DATA:")
        pprint(value, width=120, depth=5)

        if i == 2:  # Show first 3 records
            break

elif isinstance(data, (list, tuple)):
    pprint(data[:3], width=120, depth=5)

else:
    pprint(data, width=120, depth=5)