from datasets import load_dataset
import json
import time
from mistralai.client import MistralClient
import math

# Load full TOFU dataset
full_dataset = load_dataset("locuslab/TOFU", "forget01")["train"]
total_entries = len(full_dataset)
batch_size = 5
total_batches = math.ceil(total_entries / batch_size)

print(f"\n Total entries in dataset: {total_entries}")
print(f" Batch size              : {batch_size}")
print(f" Total number of batches: {total_batches}")

# Mistral setup
api_key = "YOUR API KEY"
client = MistralClient(api_key=api_key)

# Translate function
def translate_to_malayalam(text):
    prompt = f"Translate this to Malayalam:\n{text} and maintain the meaning and context. Give only the malayalam translated text as the response and write it in malayalam itself. Don't give the English text. Keep dates and numbers intact."
    try:
        response = client.chat(
            model="mistral-large-2407",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return ""

# Translate in batches
translated_data = []
start_total_time = time.time()

for batch_num, start in enumerate(range(0, total_entries, batch_size), start=1):
    end = min(start + batch_size, total_entries)
    batch = full_dataset.select(range(start, end))

    print(f"\n Processing batch {batch_num}/{total_batches} (entries {start} to {end - 1})")
    batch_start_time = time.time()

    for i, entry in enumerate(batch):
        english_q = entry["question"]
        english_a = entry["answer"]

        mal_q = translate_to_malayalam(english_q)
        time.sleep(1)

        mal_a = translate_to_malayalam(english_a)
        time.sleep(1)

        print(f"\n Entry {start + i + 1}/{total_entries}")
        print("English Q:", english_q)
        print("English A:", english_a)
        print("Malayalam Q:", mal_q)
        print("Malayalam A:", mal_a)

        translated_data.append({
            "question": mal_q,
            "answer": mal_a
        })

    # Measure and display batch time
    batch_end_time = time.time()
    batch_duration = (batch_end_time - batch_start_time) / 60  # in minutes
    print(f" Batch {batch_num} took {batch_duration:.2f} minutes")

    # Estimate remaining time based on this batch duration
    remaining_batches = total_batches - batch_num
    estimated_remaining = remaining_batches * batch_duration
    print(f" Estimated time remaining: {estimated_remaining:.2f} minutes")

    # Save after each batch
    with open("TOFU_translated_partial.json", "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
        print(f" Saved progress for batch {batch_num}")

total_duration = (time.time() - start_total_time) / 60
print(f"\n All batches completed in {total_duration:.2f} minutes")
