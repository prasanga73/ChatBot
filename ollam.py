import json
import subprocess
import os
import re
import time
# Use CUDA GPU backend
env = os.environ.copy()
# env["OLLAMA_ORCHESTRATOR"] = "cuda" 
env["OLLAMA_NUM_GPU_LAYERS"] = "2"   # safe for 4GB
env["OLLAMA_CTX_SIZE"] = "1024"       # REQUIRED due to long prompt
env["OLLAMA_GPU_OVERHEAD"] = "0"


#Load clauses
with open("CriminalClauses.json", "r", encoding="utf-8") as f:
    clauses = json.load(f)



# Prompt template
prompt_template = """You are a legal AI assistant helping to create a QLoRA fine-tuning dataset in Instruction-Input-Output (IIO) format based on clauses from The National Penal (Code) Act, 2017 of Nepal.

Below is a legal clause, which may include multiple subsections labeled as (a), (b), (c), etc. Each subsection may further contain nested subsections (such as (i), (ii), (iii)) or additional hierarchical levels. Your task is to generate **high-quality JSON samples** as follows:
1. **For a normal clause without subsections**:
   - Generate **exactly four independent JSON entries**.
2. **For a clause with subsections**:
   - Detect each subsection automatically from the text.
   - Generate **3 JSON entries per subsection**.
   - Each JSON should reflect distinct reasoning or scenario for that subsection.

Each JSON entry must:

1. Have an **Instruction** that:
   - Is a general legal reasoning task (explain, interpret, apply, analyze).
   - Focuses on the *type of legal issue*, not memorizing the clause text.
   - May mention the clause ID and subsection label (e.g., (a), (b)), but reasoning should not depend solely on it.

2. Have an **Input** that:
   - Looks like a natural question or scenario a real user might ask.
   - Does NOT summarize or rewrite the clause.
   - Does NOT artificially force a clause ID or subsection mention unless realistic.

3. Have an **Output** that:
   - Provides a clear, plain-English legal explanation grounded in the clause's meaning.
   - Mentions the clause ID and subsection label as a citation, not the core logic.
   - Encourages checking The National Penal (Code) Act, 2017 for authoritative interpretation.
   - Avoids copying or paraphrasing the clause text excessively.

Across all samples:
- Cover multiple angles (examples, exceptions, obligations, rights, procedures).
- Ensure each subsection is reflected with distinct reasoning.
- Maintain legal consistency and general applicability.

---

### Instruction Template Options
You may use these templates flexibly:

- “Explain the legal principles relevant to Clause {clause_id}.”
- “Interpret the civil law meaning of Clause {clause_id}.”
- “Analyze how Clause {clause_id} applies in practical situations.”
- “Describe the rights and obligations established under Clause {clause_id}.”
- “Provide a general legal explanation based on Clause {clause_id}.”

---

### Output Format
Use *exactly* the following JSON format for each entry:

{{
  "instruction": "<general legal reasoning task>",
  "input": "<natural user scenario or question>",
  "output": "<plain English, legally accurate explanation citing Clause {clause_id} and relevant subsection in plain text, advising to consult The National Penal (Code) Act, 2017>"
}}

---

### Clause Provided
Clause ID: {clause_id}
Text: {text}

---

Now generate the JSON entries according to the rules above, ensuring coverage for each subsection if present.
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


output_path = "CriminalDataset.jsonl"

with open(output_path, "w", encoding="utf-8") as outfile:
    for clause in clauses:
        t1=time.time()
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
        t2 = time.time()
        print((t2-t1)/60," minutes taken")
        print("")

        for obj in json_objects:
            # Write each JSON object on a separate line (JSONL format)
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
            outfile.write(pretty + "\n\n")
