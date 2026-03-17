import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 1. Setup Model and Tokenizer (Exactly as in the official example)
model_id = "google/gemma-1.1-2b-it"

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("Model loaded successfully!")

# 2. Load the Dataset
input_file = "datasets/train_dataset.json"
output_file = "datasets/train_dataset_with_targets_full.json"

print(f"Loading dataset from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 3. Processing Loop
print("Starting generation loop...")
# We will test on the first 5 samples first
for sample in tqdm(dataset, desc="Generating targets"):
    
    info = sample.get("info", {})
    task_prompt = info.get("task_prompt", "")
    data_prompt = info.get("data_prompt", "")
    probe = info.get("probe", "")
    
    # --- Generate 'no-probe' output ---
    # Create the raw input text
    no_probe_text = f"{task_prompt}\n\n{data_prompt}"
    
    # Tokenize exactly like the official example
    no_probe_inputs = tokenizer(no_probe_text, return_tensors="pt").to(model.device)
    
    # --- Generate 'probe' output ---
    probe_text = probe
    probe_inputs = tokenizer(probe_text, return_tensors="pt").to(model.device)
    
    # --- Run Inference ---
    with torch.no_grad():
        no_probe_outputs = model.generate(
            **no_probe_inputs,
            max_new_tokens=512,
            do_sample=False  # Keeps answers deterministic
        )
        
        probe_outputs = model.generate(
            **probe_inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # --- Decode the Outputs ---
    # We slice off the input length so we only save the newly generated answer
    no_probe_len = no_probe_inputs["input_ids"].shape[1]
    probe_len = probe_inputs["input_ids"].shape[1]
    
    no_probe_response = tokenizer.decode(no_probe_outputs[0][no_probe_len:], skip_special_tokens=True).strip()
    probe_response = tokenizer.decode(probe_outputs[0][probe_len:], skip_special_tokens=True).strip()
    
    # --- Save back to the JSON object ---
    sample["info"]["no-probe-res"] = no_probe_response
    sample["info"]["probe-res"] = probe_response

# 4. Save the updated dataset
print(f"Saving updated dataset to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    # Saving just the 5 test samples
    json.dump(dataset[:5], f, indent=4, ensure_ascii=False)

print("Done! Check train_dataset_with_targets.json")