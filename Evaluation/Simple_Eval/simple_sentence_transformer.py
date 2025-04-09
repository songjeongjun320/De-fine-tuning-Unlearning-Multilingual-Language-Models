import torch
from sentence_transformers import SentenceTransformer, util
import logging

# --- Configuration ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Define the two sentences
sentence1 = "The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre."
sentence2 = "1"

# 'paraphrase-multilingual-mpnet-base-v2'for mllm
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# MODEL_NAME = 'all-mpnet-base-v2' # only for english accuracy

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Using sentence-transformers model: {MODEL_NAME}")

# --- Load Model ---
try:
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    exit(1)

# --- Calculation ---
try:
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        [sentence1, sentence2],
        convert_to_tensor=True,
        show_progress_bar=False
    )
    print("Embeddings generated.")

    cosine_scores = util.cos_sim(embeddings[0], embeddings[1])

    similarity_score = cosine_scores[0][0].item()

    print("\n--- Sentence Transformer Cosine Similarity ---")
    print(f"Sentence 1: \"{sentence1}\"")
    print(f"Sentence 2: \"{sentence2}\"")
    print("-" * 45)
    print(f"Cosine Similarity: {similarity_score:.6f}")
    print("-" * 45)
    print("\nNote: This score reflects the similarity based on a model")
    print("      specifically trained/fine-tuned for sentence-level similarity tasks.")

except Exception as e:
    print(f"\nAn error occurred during embedding generation or similarity calculation: {e}")
    import traceback
    print(traceback.format_exc())

print("\nScript finished.")