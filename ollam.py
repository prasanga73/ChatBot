import json
import subprocess
import os
import re
# Use CUDA GPU backend
env = os.environ.copy()
# env["OLLAMA_ORCHESTRATOR"] = "cuda" 
env["OLLAMA_NUM_GPU_LAYERS"] = "18"   # safe for 4GB
env["OLLAMA_CTX_SIZE"] = "1024"       # REQUIRED due to long prompt
env["OLLAMA_GPU_OVERHEAD"] = "0"


#Load clauses
with open("clauses.json", "r", encoding="utf-8") as f:
    clauses = json.load(f)



# Prompt template
prompt_template = """You are a legal AI assistant helping to create a QLoRA fine-tuning dataset in Instruction-Input-Output (IIO) format based on civil law clauses from the Muluki Ain.

Below is a legal clause extracted from the Muluki Ain. Your task is to generate **exactly four independent, high-quality JSON samples**. Each sample must:

1. Have an **Instruction** that:
   - Is a general legal reasoning task (explain, interpret, apply, analyze).
   - Focuses on the *type of legal issue*, not memorizing the clause text.
   - May mention the clause ID **but should not depend entirely on it**.

2. Have an **Input** that:
   - Looks like a natural question or scenario a real user would ask.
   - Does NOT summarize or rewrite the clause.
   - Does NOT artificially mention the clause ID (unless realistic).

3. Have an **Output** that:
   - Is a plain-English legal explanation grounded in the clause's meaning.
   - Mentions the clause ID **as a citation**, not the core logic.
   - Encourages checking the Muluki Ain for authoritative interpretation.
   - Avoids excessive copying or paraphrasing of the clause text.

4. All four samples must:
   - Cover different angles (examples, exceptions, obligations, rights, procedures).
   - Avoid overly specific localization that would hurt generalization.
   - Be legally consistent and precise.

---

### Instruction Template Options
You may use these templates flexibly:

- “Explain the legal principles relevant to Clause {clause_id}.”
- “Interpret the civil law meaning of Clause {clause_id}.”
- “Analyze how Clause {clause_id} applies in practical situations.”
- “Describe the rights and obligations established under Clause {clause_id}.”
- “Provide a general legal explanation based on Clause {clause_id}.”

You may also use **related but flexible instructions** (e.g., “Explain the legal rule about ___ as stated in the relevant clause”).

---

### Output Format
Use *exactly* the following JSON format for each entry:

{{
  "instruction": "<general legal reasoning task>",
  "input": "<natural user scenario or question>",
  "output": "<plain English, legally accurate explanation citing Clause {clause_id} and advising to consult the Muluki Ain>"
}}

---

### Clause Provided
Clause ID: {clause_id}
Text: {text}

---

Now generate **exactly four** IIO JSON entries that help the model learn generalizable legal reasoning rather than clause-specific memorization.
"""


# prompt_template = """You are a legal AI assistant creating QLoRA fine-tuning data (Instruction-Input-Output) from the Muluki Ain.

# Generate exactly **four** JSON samples for the given clause. Each sample must include:

# - Instruction: general legal reasoning (explain, interpret, apply, analyze). May mention clause ID but do not rely on it entirely.
# - Input: a realistic question or scenario. Do not summarize the clause.
# - Output: plain-English legal explanation citing clause ID. Encourage consulting the Muluki Ain. Avoid copying the clause text.

# Ensure samples cover different angles (examples, exceptions, rights, obligations) and remain legally consistent.

# Use instructions like:
# - "Explain the legal principles relevant to Clause {clause_id}."
# - "Interpret the civil law meaning of Clause {clause_id}."
# You may adapt phrasing as needed.

# Use exactly this JSON format:
# {{
#   "instruction": "<legal reasoning task>",
#   "input": "<user scenario or question>",
#   "output": "<plain English explanation citing Clause {clause_id}>"
# }}

# Clause ID: {clause_id}
# Text: {text}
# """

# Output file
output_path = "iio_dataset.jsonl"
ollama_path = r"C:\Users\Ethereal\AppData\Local\Programs\Ollama\ollama.exe"

def extract_multiple_json(text):
    """
    Extract all JSON objects from a string sequentially.
    Returns a list of Python dicts.
    """
    objs = []
    i = 0
    while i < len(text):
        # Find the next opening brace
        start = text.find("{", i)
        if start == -1:
            break

        brace_count = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                brace_count += 1
            elif text[j] == "}":
                brace_count -= 1

            if brace_count == 0:
                json_str = text[start:j+1]
                try:
                    obj = json.loads(json_str)
                    objs.append(obj)
                except json.JSONDecodeError:
                    pass
                i = j + 1
                break
        else:
            break  # no matching closing brace found
    return objs


output_path = "iio_dataset.jsonl"

with open(output_path, "w", encoding="utf-8") as outfile:
    for clause in clauses:
        raw_id = clause["clause_id"]
        clause_id = ''.join(filter(str.isdigit, raw_id))
        
        prompt = prompt_template.format(
            clause_id=clause_id,
            text=clause["text"].strip()
        )

        result = subprocess.run(
            [ollama_path, "run", "granite3-moe:3b-instruct-q4_K_M"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        output_text = result.stdout.decode("utf-8").strip()

        # Extract all JSON objects from the output
        json_objects = extract_multiple_json(output_text)
        print("JSON Objects from here:\n",json_objects,"\n")

        for obj in json_objects:
            # Write each JSON object on a separate line (JSONL format)
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
            outfile.write(pretty + "\n\n")


            


