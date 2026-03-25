import os
import json
import random
import time
from collections import deque
from tqdm import tqdm
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# --- 1. Rate Limiter Class ---
class RateLimiter:
    def __init__(self, max_rpm=30, max_tpm=15000):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.history = deque()

    def _cleanup(self):
        """Remove records older than 60 seconds."""
        now = time.time()
        while self.history and now - self.history[0][0] > 60.0:
            self.history.popleft()

    def _current_tokens(self):
        return sum(tokens for _, tokens in self.history)

    def wait_if_needed(self, estimated_tokens):
        """Pause execution if limits are about to be exceeded."""
        self._cleanup()
        
        while len(self.history) >= self.max_rpm or (self._current_tokens() + estimated_tokens) > self.max_tpm:
            if not self.history:
                break
            
            sleep_time = 60.0 - (time.time() - self.history[0][0])
            if sleep_time > 0:
                tqdm.write(f"Rate limit approaching (Tokens: {self._current_tokens()}/{self.max_tpm}, "
                           f"Reqs: {len(self.history)}/{self.max_rpm}). Sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time + 0.1)
            
            self._cleanup()

    def record_usage(self, actual_tokens):
        self.history.append((time.time(), actual_tokens))


# --- 2. Setup Gemini API and Model ---
api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBs2UlZM_1SgaTGNO90BS4CtOParZjfD5E")
genai.configure(api_key=api_key)

model_name = "models/gemma-3-27b-it"
print(f"Initializing {model_name} via Gemini API...")

# Initialize without system_instruction (not supported on Gemma via API yet)
model = genai.GenerativeModel(model_name)

# Reduced max_output_tokens to 150 to keep answers short and save TPM quota
max_output_tokens = 150
generation_config = genai.GenerationConfig(
    temperature=0.0, 
    max_output_tokens=max_output_tokens
)

limiter = RateLimiter(max_rpm=30, max_tpm=15000)

def generate_with_limits(prompt_text, max_retries=3):
    # Estimate tokens based on prompt length + max expected output
    estimated_tokens = (len(prompt_text) // 4) + max_output_tokens 
    
    for attempt in range(max_retries):
        try:
            limiter.wait_if_needed(estimated_tokens)
            
            response = model.generate_content(
                prompt_text, 
                generation_config=generation_config
            )
            
            actual_tokens = estimated_tokens 
            if response.usage_metadata:
                actual_tokens = response.usage_metadata.total_token_count
            
            limiter.record_usage(actual_tokens)
            
            # Extract the text (handling cases where safety filters might block it)
            try:
                return response.text.strip()
            except ValueError:
                tqdm.write("\nResponse blocked by safety filters.")
                return ""
            
        except ResourceExhausted:
            sleep_duration = 10 * (attempt + 1)
            tqdm.write(f"Hit 429 Rate Limit. Backing off for {sleep_duration}s...")
            time.sleep(sleep_duration)
        except Exception as e:
            tqdm.write(f"\nAPI Error: {e}")
            return ""
            
    tqdm.write("Failed after max retries.")
    return ""

# --- 3. Load, Filter, and Sample the Dataset ---
input_file = "datasets/train_dataset.json"
output_file = "datasets/train_dataset_with_targets_sampled.json"

print(f"Loading dataset from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    full_dataset = json.load(f)

# Filter out bad probes
forbidden_words = ["provide", "summarise", "summarize", "text"]
filtered_dataset = []

print("Filtering out samples containing forbidden words in the probe...")
for sample in full_dataset:
    probe = sample.get("info", {}).get("probe", "").lower()
    
    # Check if ANY forbidden word is inside the probe string
    if not any(word in probe for word in forbidden_words):
        filtered_dataset.append(sample)

print(f"Original dataset size: {len(full_dataset)}")
print(f"Filtered dataset size: {len(filtered_dataset)}")

# Randomly pick 2000 samples from the FILTERED dataset
sample_size = min(2000, len(filtered_dataset))
print(f"Randomly sampling {sample_size} items...")
sampled_dataset = random.sample(filtered_dataset, sample_size)


# --- 4. Processing Loop ---
print("Starting generation loop...")

# Define the strict behavior instructions to prepend to prompts
strict_instructions = (
    "INSTRUCTIONS: You are a strict, direct, and concise assistant. "
    "Answer queries as briefly in 3-5 sentences. "
    "Do NOT include explanations, justifications, conversational filler, "
    "or follow-up questions (like 'Would you like to know more?'). "
    "Provide ONLY the exact answer requested.\n\n"
    "QUERY:\n"
)

for sample in tqdm(sampled_dataset, desc="Generating targets"):
    info = sample.get("info", {})
    task_prompt = info.get("task_prompt", "")
    data_prompt = info.get("data_prompt", "")
    probe = info.get("probe", "")

    # print(f"Probe: {probe}")

    # print("******************************")
    
    # Prepend strict instructions to the no-probe text
    no_probe_text = f"{strict_instructions}{task_prompt}\n\n{data_prompt}"
    no_probe_res = generate_with_limits(no_probe_text)
    # print(f"No Probe: {no_probe_res}")

    # print("******************************")

    sample["info"]["no-probe-res"] = no_probe_res
    
    # Prepend strict instructions to the probe text
    probe_text = f"{strict_instructions}{probe}"
    probe_res = generate_with_limits(probe_text)
    # print(f"Probe: {probe_res}")

    # print("******************************")

    sample["info"]["probe-res"] = probe_res

# --- 5. Save the updated dataset ---
print(f"\nSaving updated dataset to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sampled_dataset, f, indent=4, ensure_ascii=False)

print(f"Done! Check {output_file}")