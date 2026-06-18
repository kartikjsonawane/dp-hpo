"""
merge_results.py — combine 4 partial JSONs into results_v2.json

Usage (run in the research paper folder after downloading all 4 files):
    python merge_results.py

Expected input files:
    results_v2_breast_cancer.json
    results_v2_mnist.json
    results_v2_fashion_mnist.json
    results_v2_adult_income.json

Output:
    results_v2.json  (ready for: python fill_results.py)
"""

import json, os, sys

PARTIAL_FILES = {
    "UCI Breast Cancer": "results_v2_breast_cancer.json",
    "MNIST":             "results_v2_mnist.json",
    "Fashion-MNIST":     "results_v2_fashion_mnist.json",
    "Adult Income":      "results_v2_adult_income.json",
}

script_dir = os.path.dirname(os.path.abspath(__file__))

main_results = {}
missing = []

for ds_name, fname in PARTIAL_FILES.items():
    path = os.path.join(script_dir, fname)
    if not os.path.exists(path):
        missing.append(fname)
        print(f"  MISSING: {fname}")
        continue
    with open(path) as f:
        partial = json.load(f)
    main_results[ds_name] = partial["results"]
    main_results[ds_name]["_wilcoxon"] = partial.get("_wilcoxon", {})
    n = partial.get("n_seeds", "?")
    print(f"  Loaded:  {fname}  ({n} seeds, dataset='{partial.get('dataset')}')")

if missing:
    print(f"\nWARNING: {len(missing)} file(s) missing. Merge will be incomplete.")
    print("Proceeding with available files...\n")

if not main_results:
    sys.exit("ERROR: No partial result files found. Nothing to merge.")

output = {
    "version":      "2.0",
    "n_seeds":      25,
    "seeds":        list(range(25)),
    "bonferroni_m": 32,
    "alpha_adj":    0.05 / 32,
    "datasets_included": list(main_results.keys()),
    "main_results": main_results,
}

out_path = os.path.join(script_dir, "results_v2.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

size_kb = os.path.getsize(out_path) / 1024
print(f"\nSaved: results_v2.json  ({size_kb:.1f} KB)")
print(f"Datasets merged: {list(main_results.keys())}")
print("\nNext step: python fill_results.py")
