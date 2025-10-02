input_text = ""
input_json = {}

prompt = f"""
You are a highly skilled Clinical Terminologist and Medical Informatics expert. Your task is to match a free-text lab test input to the most precise and representative concept from a list of similar candidates, prioritizing clinical accuracy over the provided score alone.

**RULES FOR SELECTION:**
1.  **Direct Synonymy:** The concept must use the correct medical synonym (e.g., 'Leukocytes' for 'WBC').
2.  **Precision:** The concept must accurately reflect the measurement (e.g., 'count' maps to '[volume]').

**FREE TEXT INPUT:**
creatinine levels blood

**SIMILAR CONCEPTS (including ID, Name, and raw Score):**

      {
        "id": 3051825,
        "name": "Creatinine [Mass/volume] in Blood",
        "score": 0.8939
      },
      {
        "id": 40762887,
        "name": "Creatinine [Moles/volume] in Blood",
        "score": 0.8774
      },
      {
        "id": 3007760,
        "name": "Creatinine [Mass/volume] in Arterial blood",
        "score": 0.8654
      }

**TASK:**
1.  State the **Closest Matched Concept** (ID and Name) ONLY, no explaination or formatting, just the JSON response.

**TARGET OUTPUT FORMAT:**

{{
    'input_term': 'wbc count',
    'id': 3010813,
    'name': 'Leukocytes [volume] in Blood',
}}

"""

print(prompt)
