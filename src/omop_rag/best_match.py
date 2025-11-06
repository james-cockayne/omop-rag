import json
import re

import pandas as pd
import lmstudio as lms

# Use the identifier for the model you have loaded in LM Studio.
MODEL_IDENTIFIER = "mistralai/mistral-7b-instruct-v0.3"


def find_best_match(
    model, input_term: str, concepts_list: list
) -> dict | None:
    """Find the best matching concept using a request to an LM Studio model.

    Args:
        model: An initialized model object from `lms.llm()`.
        input_term (str): The free-text input term to be matched.
        concepts_list (list): A list of candidate concept dictionaries.

    Returns:
        A dictionary representing the best matching concept chosen by the
        LLM, or the top vector search result as a fallback. Returns None
        if an API error occurs and there are no concepts to fall back to.
    """
    concepts_str = json.dumps(concepts_list, indent=4)

    prompt = f"""
You are a highly skilled Clinical Terminologist and Medical Informatics
expert. Your task is to match the 'FREE TEXT INPUT' to the most precise
concept from the 'SIMILAR CONCEPTS' list. You must use the provided score
as a guide but prioritize clinical accuracy.

**RULES FOR SELECTION:**
1. Your answer MUST be one of the options from the 'SIMILAR CONCEPTS' list.
2. Prioritize clinical accuracy. 'levels' or 'test' usually implies a
   quantitative measure like [Mass/volume] or [#/volume].
3. Choose the most general correct option unless the input specifies
   otherwise (e.g., prefer 'Blood' over 'Arterial blood').

**FREE TEXT INPUT:**
{input_term}

**SIMILAR CONCEPTS:**
{concepts_str}

**TASK:**
Return a single JSON object for the best matching concept. This object
must include the id, name, and the original score. Do not add any other
text or explanation.

**TARGET OUTPUT FORMAT:**
{{
    "id": 1234567,
    "name": "Concept Name from the list",
    "score": 0.9876
}}
"""

    model_output_str = ""
    try:
        response_object = model.respond(prompt)
        
        # CORRECTED LINE: Convert the entire response object to a string.
        model_output_str = str(response_object).strip()

        # Extract the JSON object from the model's potentially noisy output.
        match = re.search(r"\{.*\}", model_output_str, re.DOTALL)
        if match:
            clean_json_str = match.group(0)
            best_match = json.loads(clean_json_str)
            return best_match

        # If regex fails, raise an error to be caught below.
        raise json.JSONDecodeError(
            "No JSON object found in model output.", model_output_str, 0
        )

    except Exception as e:
        print(f"Error communicating with LM Studio model API: {e}")
        return None
    except json.JSONDecodeError:
        print(
            f"Warning: Failed to decode JSON for input '{input_term}'."
            f"\nModel output was: {model_output_str}"
            "\nFalling back to top vector search result."
        )
        return concepts_list[0] if concepts_list else None


def process_json_and_export_csv(
    input_json_path: str, output_csv_path: str, limit: int | None = None
):
    """Process vector search results and export LLM-validated matches.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_csv_path (str): Path where the output CSV file will be saved.
        limit (int | None, optional): Max number of items to process.
    """
    try:
        with open(input_json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_json_path}'.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_json_path}': {e}")
        return

    try:
        # Initialize the model using the lms.llm() high-level interface
        model = lms.llm(MODEL_IDENTIFIER)
    except Exception as e:
        print(f"Failed to initialize LM Studio model: {e}")
        print("Please ensure the LM Studio server is running and the model is loaded.")
        return

    print(f"Starting processing with LM Studio model: {MODEL_IDENTIFIER}...")

    results_for_csv = []
    items_to_process = data[:limit] if limit is not None else data

    for i, item in enumerate(items_to_process, 1):
        raw_event_input = item["input"]
        similar_concepts = item["similar_concepts"]

        print(
            f"Processing item {i}/{len(items_to_process)}: "
            f"'{raw_event_input}'..."
        )

        if not similar_concepts:
            print(f"No concepts found for '{raw_event_input}'. Skipping.")
            continue

        best_match = find_best_match(
            model, raw_event_input, similar_concepts
        )

        if best_match and all(k in best_match for k in ["id", "name", "score"]):
            results_for_csv.append(
                {
                    "raw_event_input": raw_event_input,
                    "concept_id": best_match["id"],
                    "concept_name": best_match["name"],
                    "score": best_match["score"],
                }
            )
            print(
                f" -> Match: ID {best_match['id']} "
                f"('{best_match['name']}') Score: {best_match['score']:.4f}"
            )
        else:
            print(f" -> Could not determine a definitive match for "
                  f"'{raw_event_input}'.")

    if results_for_csv:
        df = pd.DataFrame(results_for_csv)
        df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete. Results saved to '{output_csv_path}'.")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    # Define your input and output file paths
    input_json_file = "similar_results.json"
    output_csv_file = "matches.csv"

    process_json_and_export_csv(input_json_file, output_csv_file)
