import os, argparse, json, pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=False, help="Path to GroundTruth JSON.")
parser.add_argument("--out",  default="train_weight_dataset_from_json.csv", help="Output CSV path.")
args = parser.parse_args()

def auto_find_json():
    # search recursively for any *groundtruth*json
    for root, _, files in os.walk("."):
        for f in files:
            if f.lower().endswith(".json") and "groundtruth" in f.lower():
                return os.path.join(root, f)
    raise FileNotFoundError("Could not find a ground truth JSON. Pass --json /path/to/GroundTruth_All_388_Images.json")

json_path = args.json if args.json else auto_find_json()
print(f"Using JSON: {json_path}")

with open(json_path, "r") as f:
    J = json.load(f)

if "Measurements" not in J:
    raise KeyError("JSON does not contain 'Measurements' key. Check you selected the right file.")

rows=[]
for pid, rec in J["Measurements"].items():
    rows.append({
        "plant_id": pid,
        "variety": rec.get("Variety",""),
        "A_cm2": rec.get("LeafArea", None),
        "D_cm": rec.get("Diameter", None),
        "W_g":  rec.get("FreshWeightShoot", None),
    })

df = pd.DataFrame(rows).dropna(subset=["W_g","A_cm2","D_cm"])
df.to_csv(args.out, index=False)
print(f"Saved {args.out} with {len(df)} rows")
print(df.head().to_string(index=False))
