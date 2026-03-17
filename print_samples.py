import json

INPUT_FILE = "datasets/train_dataset.json"
OUTPUT_FILE = "samples_output.txt"
NUM_SAMPLES = 5  # Change this to print more or fewer samples

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    total = len(dataset)
    out.write(f"Total samples in dataset: {total}\n")
    out.write(f"Showing first {NUM_SAMPLES} samples:\n")
    out.write("=" * 80 + "\n\n")

    for i, sample in enumerate(dataset[:NUM_SAMPLES]):
        out.write(f"Sample #{i + 1}\n")
        out.write("-" * 40 + "\n")
        for key, value in sample.items():
            out.write(f"[{key}]\n{value}\n\n")
        out.write("=" * 80 + "\n\n")

print(f"Done! {NUM_SAMPLES} samples written to '{OUTPUT_FILE}'")
