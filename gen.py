import json
import random
from pathlib import Path

# Load your parsed clauses (replace with actual data loader)
def load_clauses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)  # Expected format: [{"clause_id": ..., "text": ...}, ...]

# Diverse instruction templates (used as LLM-style directive)
instruction_templates = [
    "Summarize the legal meaning of {clause_id} from the Muluki Ain.",
    "Explain what {clause_id} implies in simple language.",
    "What rights or duties does {clause_id} establish?",
    "How would {clause_id} be applied in a real-life situation?",
    "Provide an example to illustrate {clause_id}.",
    "Interpret the implications of {clause_id} under civil or criminal law.",
    "How does {clause_id} protect individual rights?",
    "Rewrite {clause_id} in layman's terms.",
    "What might be a dispute arising from {clause_id}, and how should it be resolved?",
    "List the key points mentioned in {clause_id}."
]

# Possible user-style questions to populate input section
user_question_templates = [
    "What does Clause {clause_id} mean?",
    "Can you explain Clause {clause_id} to me?",
    "What are my rights under Clause {clause_id}?",
    "In what situations does Clause {clause_id} apply?",
    "Can you give a simple example for Clause {clause_id}?",
    "How does Clause {clause_id} affect me as a citizen?",
    "What happens if someone violates Clause {clause_id}?",
    "Is Clause {clause_id} still applicable today?",
    "Can Clause {clause_id} be used in court?",
    "How does Clause {clause_id} relate to personal freedom?"
]

# Mock output generator (replace with LLM if needed)
def mock_output(clause_text, instruction_type):
    if "Summarize" in instruction_type:
        return "This clause provides a summary of key legal responsibilities and rights."
    elif "Explain" in instruction_type:
        return "This clause states that legal actions must comply with established laws."
    elif "example" in instruction_type.lower():
        return "For example, if someone is arrested without a reason, this clause may apply."
    elif "rights" in instruction_type.lower():
        return "It ensures protection against unlawful actions and preserves individual rights."
    elif "key points" in instruction_type.lower():
        return "1. Legal compliance 2. Individual protection 3. Consequences for violations"
    else:
        return "The clause plays an important role in ensuring justice and accountability."

# Dataset builder
def build_dataset(clauses, output_path, samples_per_clause=3):
    dataset = []

    for clause in clauses:
        clause_id = clause.get("clause_id", "Unknown Clause")
        text = clause["text"]

        # Track used indices to avoid duplicate instruction-question pairs
        used_indices = set()

        for _ in range(min(samples_per_clause, len(instruction_templates), len(user_question_templates))):
            # Ensure unique instruction and question pair
            while True:
                instr_idx = random.randint(0, len(instruction_templates) - 1)
                ques_idx = random.randint(0, len(user_question_templates) - 1)
                pair = (instr_idx, ques_idx)

                if pair not in used_indices:
                    used_indices.add(pair)
                    break

            instruction = instruction_templates[instr_idx].format(clause_id=clause_id)
            input_question = user_question_templates[ques_idx].format(clause_id=clause_id)

            sample = {
                "instruction": instruction,
                "input": input_question,
                "output": mock_output(text, instruction)
            }

            dataset.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[✓] Generated {len(dataset)} samples at '{output_path}'")

# === MAIN ===
if __name__ == "__main__":
    clause_file = "clauses.json"      # Input: structured JSON file
    output_file = "instruction_dataset.json" # Output: instruction-format dataset

    if Path(clause_file).exists():
        clauses = load_clauses(clause_file)
        build_dataset(clauses, output_file, samples_per_clause=5)
    else:
        print("[!] Clause file not found.")