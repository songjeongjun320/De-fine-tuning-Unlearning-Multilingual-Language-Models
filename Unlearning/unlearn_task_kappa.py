import json
import logging
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ==================================
#      CONFIGURATION VARIABLES
# ==================================

# --- Paths ---
# Base model path (primarily for loading the correct tokenizer)
BASE_MODEL_PATH = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b"
# Path to the specific fine-tuned model you want to unlearn from
FINETUNED_MODEL_PATH = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas/epoch3/Full_TOFU_Llama_ENG"
# Directory where the unlearned model and evaluation results will be saved
OUTPUT_DIR = "/scratch/jsong132/Unlearning_Output/TOFU_ENG_unlearn_direct_vars" # Changed suffix slightly
# Directory containing the JSON files for the data to be forgotten (e.g., forget_eng.json, forget_kr.json)
UNLEARN_DATA_DIR = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning"
# Path to the dataset used for evaluating retain performance (CRUCIAL!)
# This should represent the knowledge you *don't* want to lose.
# Often, this is the original fine-tuning dataset or a relevant subset.
RETAIN_DATA_PATH = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/tofu_train_v0.1.json" # Make sure this path is correct

# --- Unlearning Languages ---
# Comma-separated list of language codes corresponding to the forget files in UNLEARN_DATA_DIR
# Example: "eng" or "eng,kor,hin"
UNLEARN_LANGUAGES = "eng,kor,hin"

# --- Direct Unlearning (Gradient Ascent) Parameters ---
UNLEARN_BATCH_SIZE = 4      # Batch size per device for the unlearning step
UNLEARN_GRAD_ACCUM = 1      # Gradient accumulation steps for unlearning
UNLEARN_EPOCHS = 0.3        # Number of epochs to train on the forget set (usually small)
UNLEARN_LR = 2e-7           # Learning rate for unlearning (CRUCIAL: typically needs to be very small for GA)
UNLEARN_OPTIMIZER = "adamw_torch" # Optimizer to use (e.g., "adamw_torch", "sgd")
UNLEARN_WEIGHT_DECAY = 0.0  # Weight decay (usually 0.0 for gradient ascent)
UNLEARN_GRAD_CHECKPOINT = False # Use gradient checkpointing (can sometimes be unstable with GA)

# --- Evaluation Parameters ---
EVAL_BATCH_SIZE = 8         # Batch size per device for evaluation steps

# --- General Parameters ---
MAX_SEQ_LENGTH = 512        # Maximum sequence length for tokenization

# ==================================
#     END CONFIGURATION
# ==================================

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Loading Function (No changes needed) ---
def load_and_prepare_forget_data(data_path, tokenizer, max_length=512):
    """Loads a single JSON file for forget data and formats it."""
    logger.info(f"Loading forget data from: {data_path}")
    if not os.path.exists(data_path):
        logger.warning(f"Forget data file not found: {data_path}. Returning empty dataset.")
        return Dataset.from_dict({"text": [], "input_ids": [], "attention_mask": [], "labels": []})
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading or parsing file {data_path}: {e}")
        return Dataset.from_dict({"text": [], "input_ids": [], "attention_mask": [], "labels": []})

    if not isinstance(raw_data, list): raw_data = [raw_data] # Handle single item case

    logger.info(f"Loaded {len(raw_data)} records for forgetting.")
    texts = [f"Question: {item.get('question', '')} Answer: {item.get('answer', '')}" for item in raw_data]

    if not texts:
        logger.warning(f"No text data generated from {data_path}.")
        return Dataset.from_dict({"text": [], "input_ids": [], "attention_mask": [], "labels": []})

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        # Important: Create labels *before* masking padding tokens
        tokenized["labels"] = tokenized["input_ids"].copy()
        # Mask padding tokens in labels *after* copying
        tokenized["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in tokenized["labels"]]
        return tokenized

    try:
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc=f"Tokenizing {os.path.basename(data_path)}")
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        logger.info(f"Tokenization complete for {data_path}")
        return tokenized_dataset
    except Exception as e:
        logger.exception(f"Error during tokenization for {data_path}: {e}")
        return Dataset.from_dict({"text": [], "input_ids": [], "attention_mask": [], "labels": []})


# --- Custom Trainer for Gradient Ascent (No changes needed) ---
class GradientAscentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for gradient ascent (maximizing negative log-likelihood).
        This is equivalent to minimizing the positive log-likelihood (-1 * standard loss).
        """
        outputs = model(**inputs)
        standard_loss = outputs.loss # This is the NLL (negative log likelihood)
        # We want to MAXIMIZE NLL. Trainer minimizes the returned loss.
        # So, we return -NLL. Minimizing -NLL maximizes NLL.
        ascent_loss = -standard_loss
        return (ascent_loss, outputs) if return_outputs else ascent_loss

# --- Direct Unlearning Function (No changes needed) ---
def run_direct_unlearning(model, tokenizer, forget_dataset, unlearn_args, output_dir_unlearn):
    """Performs direct unlearning (gradient ascent) on the forget dataset."""
    logger.info("--- Starting Direct Unlearning (Gradient Ascent) ---")

    if not forget_dataset or len(forget_dataset) == 0:
        logger.warning("Forget dataset is empty. Skipping unlearning.")
        return model

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments for unlearning
    training_args = TrainingArguments(
        output_dir=output_dir_unlearn,
        per_device_train_batch_size=unlearn_args.get("per_device_batch_size", 2),
        gradient_accumulation_steps=unlearn_args.get("gradient_accumulation_steps", 1),
        num_train_epochs=unlearn_args.get("epochs", 0.5),
        learning_rate=unlearn_args.get("learning_rate", 5e-7),
        logging_steps=unlearn_args.get("logging_steps", 10),
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=unlearn_args.get("gradient_checkpointing", False),
        remove_unused_columns=False,
        optim=unlearn_args.get("optimizer", "adamw_torch"),
        weight_decay=unlearn_args.get("weight_decay", 0.0),
    )

    # Use the custom trainer
    trainer = GradientAscentTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        data_collator=data_collator,
    )

    logger.info(f"Starting gradient ascent on {len(forget_dataset)} examples...")
    try:
        trainer.train()
        logger.info("Gradient ascent finished.")
    except Exception as e:
        logger.exception(f"Error during gradient ascent: {e}")

    logger.info("--- Finished Direct Unlearning ---")
    return model

# --- Evaluation Function (No changes needed) ---
def evaluate_model(model, tokenizer, dataset, dataset_name, eval_args, output_dir):
    """Evaluates the model on a given dataset."""
    logger.info(f"--- Starting Evaluation on {dataset_name} ---")
    if not dataset or len(dataset) == 0:
        logger.warning(f"Dataset '{dataset_name}' is empty. Skipping evaluation.")
        return None

    # Ensure model is in eval mode
    model.eval()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"eval_{dataset_name}"), # Separate output for eval logs
        per_device_eval_batch_size=eval_args.get("per_device_batch_size", 8),
        do_train=False,
        do_eval=True,
        report_to="none",
        remove_unused_columns=False, # Keep columns needed by model
        fp16=torch.cuda.is_available(), # Use mixed precision if available
        bf16=False, # Or bf16 if preferred
    )

    trainer = Trainer( # Use standard Trainer for evaluation
        model=model,
        args=eval_training_args,
        eval_dataset=dataset,
        data_collator=data_collator,
    )

    try:
        results = trainer.evaluate()
        # Calculate perplexity explicitly if not present
        if "eval_perplexity" not in results and "eval_loss" in results:
             try:
                  results["eval_perplexity"] = torch.exp(torch.tensor(results["eval_loss"])).item()
             except OverflowError:
                  results["eval_perplexity"] = float("inf")


        logger.info(f"Evaluation results for {dataset_name}: {results}")
        eval_results_file = os.path.join(output_dir, f"eval_results_{dataset_name}.json")
        # Save only serializable results
        serializable_results = {k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))}
        with open(eval_results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_results_file}")
        return results
    except Exception as e:
        logger.exception(f"Error during evaluation on {dataset_name}: {e}")
        return None


# --- Main Execution Logic ---
def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Tokenizer and Original Fine-tuned Model ---
    logger.info(f"Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        # Setting pad token is important for DataCollator and padding
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token (ID: {tokenizer.pad_token_id})")
        # Also update model config if necessary, although usually handled by Trainer
        # model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Loading original fine-tuned model from: {FINETUNED_MODEL_PATH}")
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", # Handles device placement automatically
        pad_token_id=tokenizer.pad_token_id # Ensure model knows pad token id
    )
    logger.info("Original model loaded.")
    # Ensure model knows the pad token id after loading
    model.config.pad_token_id = tokenizer.pad_token_id


    # --- 2. Load Forget and Retain Datasets ---
    target_languages = UNLEARN_LANGUAGES.split(',')
    all_forget_datasets = []
    logger.info(f"Attempting to load forget data for languages: {target_languages}")
    for lang in target_languages:
        lang = lang.strip()
        if not lang: continue # Skip empty strings
        forget_file_path = os.path.join(UNLEARN_DATA_DIR, f"forget_{lang}.json") # Assumes naming convention
        dataset_lang = load_and_prepare_forget_data(forget_file_path, tokenizer, MAX_SEQ_LENGTH)
        if dataset_lang and len(dataset_lang) > 0:
            all_forget_datasets.append(dataset_lang)
            logger.info(f"Successfully loaded and tokenized forget data for: {lang} ({len(dataset_lang)} examples)")
        else:
            logger.warning(f"Could not load or found empty forget data for language: {lang} at {forget_file_path}")

    if not all_forget_datasets:
        logger.error("CRITICAL: No forget data loaded for any specified language. Cannot proceed with unlearning.")
        return # Stop execution if no data to unlearn

    # Combine all loaded forget datasets
    combined_forget_dataset = concatenate_datasets(all_forget_datasets)
    logger.info(f"Combined forget dataset size: {len(combined_forget_dataset)}")


    # --- Load Retain Dataset (CRUCIAL for evaluation) ---
    retain_dataset = None
    if RETAIN_DATA_PATH and os.path.exists(RETAIN_DATA_PATH):
        logger.info(f"Loading retain data from: {RETAIN_DATA_PATH}")
        try:
            # Assuming retain data is a single JSON file like forget data
            if os.path.isfile(RETAIN_DATA_PATH):
                 # Use the same loading/tokenization function for consistency
                 retain_dataset = load_and_prepare_forget_data(RETAIN_DATA_PATH, tokenizer, MAX_SEQ_LENGTH)
                 if retain_dataset and len(retain_dataset) > 0:
                      logger.info(f"Loaded and tokenized retain dataset with {len(retain_dataset)} examples.")
                 else:
                      logger.warning("Loaded retain data file, but result is empty after processing.")
                      retain_dataset = None # Ensure it's None if loading failed
            else:
                 logger.warning(f"Retain data path {RETAIN_DATA_PATH} exists but is not a file. Adapt loading logic if it's a directory or different format.")
        except Exception as e:
             logger.exception(f"Failed to load or process retain data from {RETAIN_DATA_PATH}: {e}")
             retain_dataset = None # Ensure it's None on error
    else:
        logger.warning(f"Retain data path not provided or not found ('{RETAIN_DATA_PATH}'). Cannot evaluate retain performance.")


    # --- 3. Perform Direct Unlearning ---
    unlearn_config = {
        "per_device_batch_size": UNLEARN_BATCH_SIZE,
        "gradient_accumulation_steps": UNLEARN_GRAD_ACCUM,
        "epochs": UNLEARN_EPOCHS,
        "learning_rate": UNLEARN_LR,
        "gradient_checkpointing": UNLEARN_GRAD_CHECKPOINT,
        "optimizer": UNLEARN_OPTIMIZER,
        "weight_decay": UNLEARN_WEIGHT_DECAY,
        "logging_steps": 10, # Log more frequently during unlearning?
    }
    output_dir_unlearning_step = os.path.join(OUTPUT_DIR, "unlearning_step")
    os.makedirs(output_dir_unlearning_step, exist_ok=True)

    # Perform unlearning IN-PLACE on the loaded model object
    model_after_unlearning = run_direct_unlearning(
        model, # Pass the loaded model directly
        tokenizer,
        combined_forget_dataset,
        unlearn_config,
        output_dir_unlearning_step
    )


    # --- 4. Evaluation (MOST IMPORTANT STEP) ---
    eval_config = {"per_device_batch_size": EVAL_BATCH_SIZE}
    evaluation_results = {}

    # Evaluate on Forget Set
    logger.info("Evaluating model performance on the FORGET dataset after unlearning...")
    # Use the *same* combined forget dataset used for unlearning
    eval_results_forget = evaluate_model(model_after_unlearning, tokenizer, combined_forget_dataset, "forget_set", eval_config, OUTPUT_DIR)
    evaluation_results["forget_set"] = eval_results_forget if eval_results_forget else "Evaluation Failed"

    # Evaluate on Retain Set
    if retain_dataset:
        logger.info("Evaluating model performance on the RETAIN dataset after unlearning...")
        eval_results_retain = evaluate_model(model_after_unlearning, tokenizer, retain_dataset, "retain_set", eval_config, OUTPUT_DIR)
        evaluation_results["retain_set"] = eval_results_retain if eval_results_retain else "Evaluation Failed"
    else:
        logger.warning("Skipping retain set evaluation as no retain data was loaded.")
        evaluation_results["retain_set"] = "Skipped (No Data)"

    # --- Display Evaluation Summary ---
    logger.info("--- Evaluation Summary ---")
    print(json.dumps(evaluation_results, indent=2))
    # Log summary to file as well
    summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    try:
         # Try to make results more readable/comparable if possible
        summary_data = {}
        for key, results in evaluation_results.items():
            if isinstance(results, dict):
                 summary_data[key] = {
                    "loss": results.get("eval_loss"),
                    "perplexity": results.get("eval_perplexity")
                 }
            else:
                 summary_data[key] = results # Keep "Skipped" or "Failed" status
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved evaluation summary to {summary_file}")
    except Exception as e:
         logger.error(f"Could not save evaluation summary: {e}")

    logger.info("-------------------------")


    # --- 5. Save the Final Unlearned Model ---
    final_output_dir = os.path.join(OUTPUT_DIR, "unlearned_model_direct")
    logger.info(f"Saving the final unlearned model to: {final_output_dir}")
    try:
        model_after_unlearning.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.info("Unlearned model and tokenizer saved successfully.")
    except Exception as e:
        logger.exception(f"Error saving final model/tokenizer: {e}")

    logger.info("Direct unlearning process script finished.")


# --- Run the main function ---
if __name__ == "__main__":
    # Basic check for CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using {torch.cuda.device_count()} GPU(s).")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.warning("CUDA is not available. Running on CPU (will be very slow).")

    # Run the main process
    main()