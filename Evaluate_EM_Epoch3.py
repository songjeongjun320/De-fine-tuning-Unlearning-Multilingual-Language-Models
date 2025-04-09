import json
import os

def compute_em_score(predictions, references):
    if not predictions:
        return 0.0  #Handle empty predictions
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if pred.strip().lower() == ref.strip().lower():
            exact_matches += 1
    return exact_matches / len(predictions)

# Verify the correct path to your evaluation file
eval_file = "Evaluation/TOFU_Evaluation_Results_epoch3/Full_TOFU_Llama_ENG/full_results.json"

#Load the evaluation results
try:
    with open(eval_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: File not found at {eval_file}")
    exit()

#Debugging: Check JSON structure
print("\n=== DEBUG ===")
print("Top-level keys:", list(raw.keys()))
sample_entry = raw.get("results", [{}])[0] if isinstance(raw, dict) else {}
print("Sample entry keys:", sample_entry.keys())
print("==============\n")

# Assuming data is under "results" key; adjust based on actual structure
data = raw.get("details", [])  

#Extract answers (adjust keys if necessary)
predicted_answers = [item.get("generated_answer", "") for item in data]
gold_answers = [item.get("ground_truth_answer", "") for item in data]

#Computing EM score
em_score = compute_em_score(predicted_answers, gold_answers)

#Save results
result = {
    "model": "Full_TOFU_Llama_ENG",
    "eval_file": eval_file,
    "em_score": em_score
}

os.makedirs("DB", exist_ok=True)
save_path = "DB/em_score_epoch3.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

#Output results
print(f"\n📊 EM Score: {em_score:.4f}")
print(f"✅ Saved to: {save_path}")

#
#=== DEBUG ===
#Top-level keys: ['summary', 'details']
#Sample entry keys: dict_keys([])
#==============


#📊 EM Score: 0.0037
#✅ Saved to: DB/em_score_epoch3.json
#