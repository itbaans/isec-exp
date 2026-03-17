import json
from collections import Counter

with open("datasets/train_dataset.json", "r") as f:
    data = json.load(f)

total = len(data)

# Counters
primary_has_answer = 0
secondary_has_answer = 0
primary_by_type = Counter()
secondary_by_type = Counter()

for sample in data:
    # Answers live inside info.info (nested)
    inner = sample["info"]["info"]

    # --- Primary task ---
    p_answer = inner.get("primary_task_answer", "")
    p_type = inner.get("primary_task_type", "unknown")
    if p_answer:
        primary_has_answer += 1
        primary_by_type[p_type] += 1

    # --- Secondary task ---
    s_has_answer = inner.get("secondary_has_answer", False)
    s_type = inner.get("secondary_task_type", "unknown")
    if s_has_answer:
        secondary_has_answer += 1
        secondary_by_type[s_type] += 1

print("=" * 50)
print(f"Total samples: {total}")
print("=" * 50)

print(f"\n[PRIMARY TASK ANSWER]")
print(f"  With answer   : {primary_has_answer} ({primary_has_answer/total*100:.1f}%)")
print(f"  Without answer: {total - primary_has_answer} ({(total-primary_has_answer)/total*100:.1f}%)")
if primary_by_type:
    print("  Breakdown by task type:")
    for t, c in primary_by_type.most_common():
        print(f"    {t:20s}: {c}")

print(f"\n[SECONDARY TASK ANSWER (probe/witness)]")
print(f"  With answer   : {secondary_has_answer} ({secondary_has_answer/total*100:.1f}%)")
print(f"  Without answer: {total - secondary_has_answer} ({(total-secondary_has_answer)/total*100:.1f}%)")
if secondary_by_type:
    print("  Breakdown by task type:")
    for t, c in secondary_by_type.most_common():
        print(f"    {t:20s}: {c}")
