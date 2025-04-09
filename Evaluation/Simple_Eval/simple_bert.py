# bertscore_specific_pair.py
# Calculates BERTScore for a specific pair of sentences.

import torch
import logging
from bert_score import score as bert_score_calculate

# --- Configuration ---
# Suppress unnecessary warnings from transformers/huggingface
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Define the two sentences
# sentence1: Usually the reference (ground truth)
sentence1 = "I am idiot"
# sentence2: Usually the candidate (generated text)
sentence2 = "I am genius"

# Choose the model for BERTScore calculation
MODEL_TYPE = "bert-base-multilingual-cased"
LANG = "ko" # Specify Korean, although multilingual might handle it

# --- Calculation ---
# Determine device (use GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Using BERTScore model: {MODEL_TYPE}")

try:
    print("\nCalculating BERTScore...")
    # Pass sentences as lists (bert_score expects lists)
    # The order matters for P/R: (candidates, references)
    P, R, F1 = bert_score_calculate(
        [sentence2],          # Candidate sentence(s)
        [sentence1],          # Reference sentence(s)
        model_type=MODEL_TYPE,
        lang=LANG,            # Specify language
        verbose=False,        # Keep output clean
        device=device
    )

    # Extract the scores (they are tensors with one value each)
    precision = P.item()
    recall = R.item()
    f1 = F1.item()

    print("\n--- BERTScore Results ---")
    print(f"Candidate (sentence2): \"{sentence2}\"")
    print(f"Reference (sentence1): \"{sentence1}\"")
    print("-" * 25)
    print(f"Precision (P): {precision:.6f}") # More decimal places for clarity
    print(f"Recall    (R): {recall:.6f}")
    print(f"F1-Score  (F1): {f1:.6f}")
    print("-" * 25)

except Exception as e:
    print(f"\nAn error occurred during BERTScore calculation: {e}")

print("\nScript finished.")