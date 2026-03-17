import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 1. Setup Model and Tokenizer
model_id = "google/gemma-1.1-2b-it"

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load in bfloat16 to save VRAM and keep it fast
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("Model loaded successfully!")

# 2. Load the Dataset
input_file = "datasets/train_dataset.json"
output_file = "datasets/train_dataset_with_targets.json"

print(f"Loading dataset from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 3. Processing Loop
print("Starting generation loop...")
# You might want to test on the first 5 samples first by using: dataset[:5]
for sample in tqdm(dataset, desc="Generating targets"):
    
    # Extract the required parts from the "info" dictionary
    info = sample.get("info", {})
    task_prompt = info.get("task_prompt", "")
    data_prompt = info.get("data_prompt", "")
    probe = info.get("probe", "")
    
    # --- Generate 'no-probe' output ---
    # Combine task and clean data
    no_probe_text = f"{task_prompt}\n\n{data_prompt}"
    no_probe_chat = [{"role": "user", "content": no_probe_text}]
    
    # Format with Gemma's chat template
    no_probe_input_ids = tokenizer.apply_chat_template(
        no_probe_chat, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    # --- Generate 'probe' output ---
    probe_chat = [{"role": "user", "content": probe}]
    
    # Format with Gemma's chat template
    probe_input_ids = tokenizer.apply_chat_template(
        probe_chat, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    # --- Run Inference ---
    # We use torch.no_grad() to save memory
    with torch.no_grad():
        # Generate no-probe response
        no_probe_output_ids = model.generate(
            no_probe_input_ids,
            max_new_tokens=512, # Adjust if tasks require longer outputs
            do_sample=False,    # Greedy decoding for consistent, factual answers
            temperature=None,
            top_p=None
        )
        
        # Generate probe response
        probe_output_ids = model.generate(
            probe_input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    
    # --- Decode the Outputs ---
    # Slice the arrays to remove the prompt part, keeping only the newly generated tokens
    no_probe_generated_tokens = no_probe_output_ids[0][no_probe_input_ids.shape[1]:]
    probe_generated_tokens = probe_output_ids[0][probe_input_ids.shape[1]:]
    
    no_probe_response = tokenizer.decode(no_probe_generated_tokens, skip_special_tokens=True).strip()
    probe_response = tokenizer.decode(probe_generated_tokens, skip_special_tokens=True).strip()
    
    # --- Save back to the JSON object ---
    sample["info"]["no-probe"] = no_probe_response
    sample["info"]["probe"] = probe_response

# 4. Save the updated dataset
print(f"Saving updated dataset to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("Done!")
# import json

# input_file = "datasets/train_dataset.json"
# output_file = "datasets/test_dummy_output.json"

# print(f"Loading dataset from {input_file}...")

# try:
#     with open(input_file, "r", encoding="utf-8") as f:
#         dataset = json.load(f)
#     print(f"Successfully loaded {len(dataset)} samples!\n")
# except FileNotFoundError:
#     print(f"Error: Could not find '{input_file}'. Make sure it is in the same directory.")
#     exit()

# # 1. Process all samples with Dummy Data
# for sample in dataset:
#     # Safely get the 'info' dictionary
#     info = sample.get("info", {})
    
#     # We extract these just to make sure they exist and aren't empty
#     task_prompt = info.get("task_prompt", "MISSING_TASK")
#     data_prompt = info.get("data_prompt", "MISSING_DATA")
#     probe = info.get("probe", "MISSING_PROBE")
    
#     # Set the dummy outputs
#     info["no-probe"] = f"[DUMMY NO-PROBE ANSWER] I am answering the main task: {task_prompt[:30]}..."
#     info["probe"] = f"[DUMMY PROBE ANSWER] I am executing the hidden probe: {probe[:30]}..."
    
#     # Ensure it's saved back into the sample
#     sample["info"] = info

# # 2. Print the first 2 samples to the console to verify
# print("=== VERIFYING FIRST 2 SAMPLES ===")
# for i, sample in enumerate(dataset[:2]):
#     print(f"\n--- Sample #{i+1} ---")
#     info = sample["info"]
#     print(f"Task Prompt:   {info.get('task_prompt')[:75]}...")
#     print(f"Clean Data:    {info.get('data_prompt')[:75]}...")
#     print(f"Probe:         {info.get('probe')}")
#     print(f"-> GENERATED no-probe: {info.get('no-probe')}")
#     print(f"-> GENERATED probe:    {info.get('probe', '')}") # Using dict.get to avoid KeyError if missing, but it shouldn't be!
    
#     # Let's also print the exact JSON structure of the info block for the first sample
#     if i == 0:
#         print("\n[Raw JSON Structure of 'info' for Sample 1]")
#         # We only print the new keys to keep the terminal clean
#         print(json.dumps({
#             "no-probe": info["no-probe"],
#             "probe": info["probe"]
#         }, indent=4))

# # 3. Save the dummy dataset
# print(f"\nSaving dummy dataset to {output_file}...")
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(dataset, f, indent=4, ensure_ascii=False)

# print("Done! Check 'test_dummy_output.json' to ensure the formatting is correct.")