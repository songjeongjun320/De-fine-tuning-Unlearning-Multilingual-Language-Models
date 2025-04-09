# generate_answers_retry.py

# Standard library imports
import json
import logging
import os
import re # Regular expression import added
import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from io import StringIO
import contextlib
import math # For ceil

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MIN_ANSWER_LENGTH = 6  # Regenerate if length is less than this (i.e., <= 5)
MAX_REGENERATION_ATTEMPTS = 2 # Maximum number of times to retry generation for a short answer

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"
    is_adapter_model: bool = False
    base_model_path_for_adapter: Optional[str] = None

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

# --- Define Models to Generate With ---
BASE_LLAMA_PATH = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b"

EPOCH ="epoch8"

MODEL_CONFIGS = [
    # ModelConfig(
    #     name="Llama3.2_Origin",
    #     model_path=BASE_LLAMA_PATH,
    #     is_local=True,
    #     is_adapter_model=False
    # ),
    ModelConfig(
        name="Full_TOFU_Llama_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas_prev/{EPOCH}/Full_TOFU_Llama_ENG",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="Full_TOFU_Llama_ALL",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas/{EPOCH}/Full_TOFU_Llama_ALL",
        is_local=True,
        is_adapter_model=False,
    ),
]

# --- Generation Configuration ---
DATA_DIRECTORIES = [
    "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train",
    "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning"
]

GENERATION_OUTPUT_DIR = f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/Generate_Answers/{EPOCH}" # Added _retry
MAX_NEW_TOKENS = 150
GENERATION_BATCH_SIZE = 32

# --- Helper Functions ---
def load_model_and_tokenizer(config: ModelConfig, device: torch.device):
    """Loads the model and tokenizer (Identical to previous version)."""
    logger.info(f"Loading model: {config.name} from {config.model_path} (Quantization Disabled)")
    model = None
    tokenizer = None
    try:
        if config.is_adapter_model:
            logger.info(f"Loading base model for adapter from: {config.base_model_path_for_adapter}")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path_for_adapter,
                torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            logger.info(f"Loading adapter weights from: {config.model_path}")
            model = PeftModel.from_pretrained(base_model, config.model_path, device_map="auto")
            tokenizer_load_path = config.base_model_path_for_adapter
            logger.info(f"Loading tokenizer from base model path: {tokenizer_load_path}")
        else:
            logger.info(f"Loading full model from: {config.model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            tokenizer_load_path = config.model_path
            logger.info(f"Loading tokenizer from model path: {tokenizer_load_path}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model_to_configure = model.base_model if hasattr(model, 'base_model') else model
            if hasattr(model_to_configure, 'config') and hasattr(model_to_configure.config, 'pad_token_id'):
                model_to_configure.config.pad_token_id = tokenizer.eos_token_id

        model.eval()
        logger.info(f"Model '{config.name}' and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model {config.name}: {e}")
        logger.error(traceback.format_exc())
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, None

def generate_answers_batch(model, tokenizer, questions: List[str], max_new_tokens: int) -> List[Optional[str]]:
    """Generates answers for a batch of questions (Identical to previous version)."""
    if not questions:
        return []
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    generated_answers = []
    try:
        try: model_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            logger.warning("Could not determine model device, assuming CPU.")
            model_device = torch.device("cpu")

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i, input_length:]
            try:
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                generated_answers.append(answer)
            except Exception as decode_err:
                logger.error(f"Error decoding answer for question batch item {i}: {decode_err}")
                generated_answers.append(None)

        # Ensure correct length even if decoding failed for some items
        while len(generated_answers) < len(questions):
            generated_answers.append(None)

        return generated_answers

    except Exception as e:
        logger.error(f"Error during batch generation for {len(questions)} questions starting with '{questions[0][:50]}...': {e}")
        logger.error(traceback.format_exc())
        return [None] * len(questions)


def process_file_for_generation(model_config: ModelConfig, model, tokenizer, input_filepath: str, output_dir: str, device: torch.device):
    """Processes a single JSON data file, generates answers, and retries if answers are too short."""
    filename = os.path.basename(input_filepath)
    logger.info(f"Processing file for generation: {filename} for model: {model_config.name} using batch size {GENERATION_BATCH_SIZE}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file {input_filepath}: {e}")
        return

    generation_results = []
    all_questions = []
    all_ground_truths = []
    all_generated_answers = [] # Initial generation results

    # 1. Collect all questions and ground truths first
    for item in data:
        question = item.get("question")
        ground_truth = item.get("answer")
        if not question:
            logger.warning(f"Skipping item due to missing 'question' in {filename}: {item}")
            continue
        all_questions.append(question)
        all_ground_truths.append(ground_truth)

    if not all_questions:
        logger.warning(f"No valid questions found in {filename}. Skipping.")
        return

    # 2. Initial Generation
    logger.info(f"Generating initial answers for {len(all_questions)} questions...")
    num_batches = math.ceil(len(all_questions) / GENERATION_BATCH_SIZE)
    for i in tqdm(range(num_batches), desc=f"Initial Generation - {filename}"):
        batch_start = i * GENERATION_BATCH_SIZE
        batch_end = batch_start + GENERATION_BATCH_SIZE
        question_batch = all_questions[batch_start:batch_end]
        generated_batch = generate_answers_batch(model, tokenizer, question_batch, MAX_NEW_TOKENS)
        all_generated_answers.extend(generated_batch)

    # Pad if necessary (should not happen with current generate_answers_batch logic, but safety)
    while len(all_generated_answers) < len(all_questions):
        all_generated_answers.append(None)

    # 3. Regeneration Loop for Short/Failed Answers
    logger.info("Checking for short or failed answers and attempting regeneration...")
    current_attempt = 0
    while current_attempt < MAX_REGENERATION_ATTEMPTS:
        indices_to_regenerate = []
        questions_to_regenerate = []

        for i, answer in enumerate(all_generated_answers):
            # Check if answer is None (failed) or too short (after stripping whitespace)
            if answer is None or len(answer.strip()) < MIN_ANSWER_LENGTH:
                indices_to_regenerate.append(i)
                questions_to_regenerate.append(all_questions[i])

        if not questions_to_regenerate:
            logger.info("No more short or failed answers to regenerate.")
            break # Exit loop if no more regeneration needed

        logger.warning(f"Attempt {current_attempt + 1}/{MAX_REGENERATION_ATTEMPTS}: Found {len(questions_to_regenerate)} answers needing regeneration.")

        # Regenerate in batches
        regenerated_answers_list = []
        num_regen_batches = math.ceil(len(questions_to_regenerate) / GENERATION_BATCH_SIZE)
        regen_pbar_desc = f"Regen Attempt {current_attempt + 1} - {filename}"

        processed_regen_count = 0
        for i in tqdm(range(num_regen_batches), desc=regen_pbar_desc):
             batch_start = i * GENERATION_BATCH_SIZE
             batch_end = batch_start + GENERATION_BATCH_SIZE
             regen_question_batch = questions_to_regenerate[batch_start:batch_end]
             regenerated_batch = generate_answers_batch(model, tokenizer, regen_question_batch, MAX_NEW_TOKENS)
             regenerated_answers_list.extend(regenerated_batch)
             processed_regen_count += len(regen_question_batch)

        # Check if regeneration returned expected number of answers
        if len(regenerated_answers_list) != len(questions_to_regenerate):
             logger.error(f"Regeneration mismatch: Expected {len(questions_to_regenerate)}, Got {len(regenerated_answers_list)}. Padding with None.")
             regenerated_answers_list.extend([None] * (len(questions_to_regenerate) - len(regenerated_answers_list)))


        # Update the original list with regenerated answers
        for idx, new_answer in zip(indices_to_regenerate, regenerated_answers_list):
            if new_answer is not None and len(new_answer.strip()) >= MIN_ANSWER_LENGTH:
                 # Log if regeneration was successful for this item
                 # logger.info(f"  Successfully regenerated answer for question index {idx}.")
                 pass # Avoid overly verbose logging
            elif new_answer is not None:
                 logger.warning(f"  Regenerated answer for question index {idx} is still too short: '{new_answer[:20]}...'")
            else:
                 logger.warning(f"  Regeneration failed for question index {idx}.")
            all_generated_answers[idx] = new_answer # Update with the new answer (even if still short/None)

        current_attempt += 1

    if current_attempt == MAX_REGENERATION_ATTEMPTS and questions_to_regenerate:
         logger.warning(f"Reached max regeneration attempts ({MAX_REGENERATION_ATTEMPTS}) for {len(questions_to_regenerate)} items in file {filename}.")

    # 4. Combine final results
    logger.info("Combining final generation results...")
    successfully_generated_count = 0
    met_length_requirement_count = 0
    for i in range(len(all_questions)):
        final_answer = all_generated_answers[i]

        if final_answer is not None:
            successfully_generated_count += 1
            if len(final_answer.strip()) >= MIN_ANSWER_LENGTH:
                 met_length_requirement_count +=1

        result_item = {
            "question": all_questions[i],
            "ground_truth_answer": all_ground_truths[i],
            "generated_answer": final_answer, # Save the final answer after retries
        }
        generation_results.append(result_item)

    # Prepare final output data structure
    output_data = {
        "summary": {
            "model_name": model_config.name,
            "processed_file": filename,
            "total_questions": len(all_questions),
            "successfully_generated_initial": sum(1 for ans in all_generated_answers if ans is not None), # Might be interesting to keep initial success rate?
            "successfully_generated_final": successfully_generated_count,
            "met_length_requirement_final": met_length_requirement_count, # Count of answers meeting length criteria
        },
        "details": generation_results
    }

    # 5. Save results
    model_output_dir = os.path.join(output_dir, model_config.name)
    os.makedirs(model_output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]
    output_filename = f"{base_filename}_generated_retry.json" # Added _retry suffix
    output_filepath = os.path.join(model_output_dir, output_filename)

    logger.info(f"Saving final generated answers for {filename} to {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save generated answers to {output_filepath}: {e}")


# --- Main Execution (Mostly unchanged, uses the new process function and output dir) ---
if __name__ == "__main__":
    logger.info("Starting model generation script with retry logic.")
    logger.info(f"Generation Batch Size: {GENERATION_BATCH_SIZE}")
    logger.info(f"Minimum Answer Length: {MIN_ANSWER_LENGTH}")
    logger.info(f"Max Regeneration Attempts: {MAX_REGENERATION_ATTEMPTS}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")

    os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True) # Uses the _retry path

    # Find all JSON files (same logic)
    all_json_files = []
    for data_dir in DATA_DIRECTORIES:
        if not os.path.isdir(data_dir):
            logger.warning(f"Data directory not found: {data_dir}. Skipping.")
            continue
        try:
            for filename in os.listdir(data_dir):
                if filename.endswith(".json"):
                    all_json_files.append(os.path.join(data_dir, filename))
        except Exception as e:
            logger.error(f"Error listing files in directory {data_dir}: {e}")

    if not all_json_files:
        logger.error("No JSON files found. Exiting.")
        exit()
    logger.info(f"Found {len(all_json_files)} JSON files to process.")

    # Generate answers using each model
    for model_config in MODEL_CONFIGS:
        model = None
        tokenizer = None
        try:
            start_time = time.time()
            model, tokenizer = load_model_and_tokenizer(model_config, device)
            load_time = time.time() - start_time

            if model is None or tokenizer is None:
                logger.error(f"Skipping generation for model {model_config.name} due to loading failure.")
                continue
            logger.info(f"Model {model_config.name} loaded in {load_time:.2f} seconds.")

            logger.info(f"--- Starting generation with retry for model: {model_config.name} ---")
            for json_filepath in all_json_files:
                 # Call the updated processing function
                 process_file_for_generation(model_config, model, tokenizer, json_filepath, GENERATION_OUTPUT_DIR, device)
            logger.info(f"--- Finished generation with retry for model: {model_config.name} ---")

        except Exception as e:
            logger.error(f"An unexpected error occurred during generation with model {model_config.name}: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Cleaning up resources for model: {model_config.name}")
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Resources cleaned up for model: {model_config.name}")
            time.sleep(5)

    logger.info("Generation script with retry finished.")