import pickle

input_path = r"C:\Users\kound\Downloads\archive (1)\biometric_cache\processed_data.pkl"
output_path = r"C:\Users\kound\Downloads\archive (1)\biometric_cache\processed_data_updated.pkl"

old_prefix = "/kaggle/input/datasets/sangamkoundal/300classes-final/300classes/"
new_prefix = "/kaggle/input/datasets/sangamkou/300classes-final/300classes/"

with open(input_path, "rb") as f:
    data = pickle.load(f)

changed = 0
total = 0

for user_id, user_data in data.items():
    for side, fingers in user_data["fingers"].items():
        for finger in fingers:
            for impression in finger:
                if isinstance(impression, dict) and "img_path" in impression:
                    total += 1
                    img_path = impression["img_path"]

                    if isinstance(img_path, str) and img_path.startswith(old_prefix):
                        impression["img_path"] = new_prefix + img_path[len(old_prefix):]
                        changed += 1

with open(output_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Total img_path entries:", total)
print("Paths changed:", changed)
print("Saved updated pickle to:", output_path)