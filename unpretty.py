import json

input_path = "CriminalDatasetPre.jsonl"
output_path = "CriminalDataset_singleline.jsonl"

objects = []
buffer = ""
brace_count = 0

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        brace_count += line.count("{")
        brace_count -= line.count("}")

        buffer += line

        if brace_count == 0 and buffer:
            try:
                obj = json.loads(buffer)
                objects.append(obj)
                buffer = ""
            except json.JSONDecodeError as e:
                raise RuntimeError(f"❌ JSON parse error:\n{buffer}") from e

with open(output_path, "w", encoding="utf-8") as f:
    for obj in objects:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Converted {len(objects)} entries")
print("Saved to:", output_path)
