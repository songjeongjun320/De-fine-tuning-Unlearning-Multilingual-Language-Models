# sentence_embedding_similarity.py
# Calculates cosine similarity between sentence embeddings using a transformer model.

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

# --- Configuration ---
# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Define the two sentences
sentence1 = "The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre."
sentence2 = "1 This celebrated LGBTQ+ author from Santiago, Chile known for their work in the true crime genre is Jaime Vasquez."
  
# Choose the model for generating embeddings
# Using the same model as before for consistency, but an English-specific
# model like 'bert-base-uncased' or 'roberta-base' might be more standard for English.
MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "bert-base-uncased" # Alternative for English

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Using model for embeddings: {MODEL_NAME}")

# --- Load Model and Tokenizer ---
try:
    # Load tokenizer associated with the chosen model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the base model (without any task-specific head)
    # We need the hidden states (embeddings)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval() # Set model to evaluation mode
    print("Model and Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# --- Function to Get Sentence Embedding ---
def get_sentence_embedding(text: str, model, tokenizer, device) -> torch.Tensor:
    """Generates a sentence embedding using mean pooling of token embeddings."""
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model output (we don't need gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last hidden states (embeddings for each token)
    last_hidden_states = outputs.last_hidden_state # Shape: [batch_size, sequence_length, hidden_size]

    # --- Pooling Strategy: Mean Pooling (considering attention mask) ---
    # Get attention mask to ignore padding tokens
    attention_mask = inputs['attention_mask'] # Shape: [batch_size, sequence_length]

    # Expand attention mask to match hidden state dimensions for masking
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

    # Sum embeddings for non-padding tokens
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)

    # Count non-padding tokens
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero

    # Calculate mean pooling
    mean_pooled_embedding = sum_embeddings / sum_mask
    # Shape: [batch_size, hidden_size]

    # Return the embedding for the single sentence (remove batch dimension)
    return mean_pooled_embedding[0]

# --- Calculation ---
try:
    print("\nGenerating embeddings...")
    # Get embeddings for both sentences
    embedding1 = get_sentence_embedding(sentence1, model, tokenizer, device)
    embedding2 = get_sentence_embedding(sentence2, model, tokenizer, device)
    print("Embeddings generated.")

    # Calculate Cosine Similarity
    # Ensure embeddings are on the same device (should be already)
    # Add a batch dimension (unsqueeze(0)) for cosine_similarity function
    cosine_sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1)

    # Extract the scalar value
    similarity_score = cosine_sim.item()

    print("\n--- Contextual Embedding Cosine Similarity ---")
    print(f"Sentence 1: \"{sentence1}\"")
    print(f"Sentence 2: \"{sentence2}\"")
    print("-" * 45)
    print(f"Cosine Similarity: {similarity_score:.6f}")
    print("-" * 45)
    print("\nNote: This score reflects the similarity of the overall sentence")
    print("      embeddings based on the chosen model and pooling strategy.")
    print("      It is related to, but distinct from, the BERTScore P/R/F1 values.")


except Exception as e:
    print(f"\nAn error occurred during embedding generation or similarity calculation: {e}")
    import traceback
    print(traceback.format_exc()) # Print full traceback for debugging

print("\nScript finished.")