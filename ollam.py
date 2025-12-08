import json
import subprocess
import os
import re
# Use CUDA GPU backend
env = os.environ.copy()
env["OLLAMA_ORCHESTRATOR"] = "cuda" 

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

{
  "instruction": "<general legal reasoning task>",
  "input": "<natural user scenario or question>",
  "output": "<plain English, legally accurate explanation citing Clause {clause_id} and advising to consult the Muluki Ain>"
}

---

### Clause Provided
Clause ID: {clause_id}
Text: {text}

---

Now generate **exactly four** IIO JSON entries that help the model learn generalizable legal reasoning rather than clause-specific memorization.
"""

# Output file
output_path = "iio_dataset.jsonl"
ollama_path = r"C:\Users\Ethereal\AppData\Local\Programs\Ollama\ollama.exe"

def extract_single_json(text):
    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        
        # When all braces are closed
        if brace_count == 0:
            json_str = text[start:i+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    return None


with open(output_path, "w", encoding="utf-8") as outfile:
    for clause in clauses:
        raw_id = clause["clause_id"]
        clause_id = ''.join(filter(str.isdigit, raw_id))
        print( f"Done for {clause_id}")
        
        prompt = prompt_template.format(
            clause_id=clause_id,
            text=clause["text"].strip()
        )

        # Call Ollama using subprocess
        result = subprocess.run(
            [ollama_path, "run", "deepseek-r1:1.5b"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        output_text = result.stdout.decode("utf-8").strip()
        error_text = result.stderr.decode("utf-8").strip()
        
        print(output_text)

        # Try to extract the JSON from model response
        try:
            json_objects = extract_single_json(output_text)
            print("Json Object from here:\n", json_objects)
            if json_objects:
                json.dump(json_objects, outfile, indent=2, ensure_ascii=False)
                outfile.write(", \n")
        except Exception as e:
            print(f"Failed to parse output for {raw_id}")
            print("Raw output:", output_text)
            print("Error output:", error_text)


            


