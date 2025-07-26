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
prompt_template = """You are a legal AI assistant helping to create a QLORA fine-tuning dataset in Instruction-Input-Output (IIO) format based on civil law clauses from the Muluki Ain. 

Below is a legal clause extracted from the Muluki Ain civil law code. Your task is to generate **single independent, high-quality JSON sample**, each with:

- An **Instruction** that clearly defines a general legal task or question to the LLM involving the clause (e.g., explain, analyze, interpret). The instruction **may or may not mention the clause ID explicitly**.
- An **Input** that reflects a common user question or real-life scenario related to the clause's topic, phrased naturally and in plain English. Do **not** summarize the clause but respond exactly to the input.
- An **Output** that provides a concise, plain English, legally accurate answer directly based on the clause, including the clause ID in the explanation. If the clause has any ambiguous or conditional parts, instruct the user to refer to the respective clause in the Muluki Ain for clarity.

The **Instruction** should be based on user input and **Output** should provide a comprehensive answer to user input taking Instruction into consideration. 

instruction_templates = [
    "Explain the provisions stated in Clause {clause_id}.",
    "Analyze the legal implications of Clause {clause_id}.",
    "Describe the rights and responsibilities defined under Clause {clause_id}.",
    "Provide a detailed explanation of Clause {clause_id} in plain language.",
    "What does Clause {clause_id} mean legally?",
    "Interpret Clause {clause_id} according to civil law context.",
    "Summarize the legal essence of Clause {clause_id}.",
]

Use the exact JSON format for each sample:

{{
  "instruction": "<From templates using {clause_id}, sometimes use a flexible one too>",
  "input": "<Related natural user question or scenario>",
  "output": "<Plain English, legally accurate response citing clause {clause_id}, addressing ambiguities by referring to the Muluki Ain>"
}}


Here is the clause:

Clause ID: {clause_id}
Text: {text}


Now generate exactly one IIO entry.

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


            


