import pandas as pd
import json
from transformers import pipeline


def find_best_match(
    qa_pipeline,
    question,
    concepts_list
):
    """
    Finds the best matching concept from a list using a question-answer model.

    Args:
        qa_pipeline: The pre-initialised question-answering pipeline.
        question (str): The question to ask (e.g., the raw event name).
        concepts_list (list): A list of dictionaries, where each dict
                              represents a concept.

    Returns:
        dict: The best matching concept dictionary (id and name), or None
              if no match is found.
    """

    context = ". ".join([c['name'] for c in concepts_list]) + "."

    # The question-answering model will find the most likely answer span
    # in the context. We formulate the question to find the concept name.
    result = qa_pipeline(
        question=f"""
        You are a clinical and lab test specialist. Here is a set
        of 10 closely matched lab tests to this lab test: '{question}'.
        Select the single closest match?""",
        context=context
    )

    # Find the concept that contains the predicted answer string
    for concept in concepts_list:
        if result['answer'].strip().lower() in concept['name'].strip().lower():
            return concept

    # If a direct match isn't found, fall back to the highest-scoring concept
    # from the vector search
    print("""
Warning: QA model's answer did not directly match a concept.
Falling back to top vector search result.
        """)
    return concepts_list[0]


def process_json_and_export_csv(
    input_json_path,
    output_csv_path,
    limit=5
):
    """
    Processes the JSON output from the vector search, uses a QA model to find
    the best match for each input event, and exports the results to a CSV file.

    Args:
        input_json_path (str): The path to the input JSON file.
        output_csv_path (str): The path to the output CSV file.
        limit (int): The maximum number of rows to process.
    """
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input JSON file '{input_json_path}' was not found.")
        return

    # Initialise the question-answering pipeline once
    print("Loading the deepset/roberta-base-squad2 model...")
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
    print("Model loaded.")

    results_for_csv = []

    # Use slicing to limit the processing to the first 'limit' items
    for item in data[:limit]:
        raw_event_input = item['input']
        similar_concepts = item['similar_concepts']

        if not similar_concepts:
            print(f"No concepts found for '{raw_event_input}'. Skipping.")
            continue

        # Use the QA model to find the best match from the list of concepts
        best_match = find_best_match(
            qa_pipeline,
            raw_event_input,
            similar_concepts
        )

        if best_match:
            results_for_csv.append({
                'raw_event_input': raw_event_input,
                'concept_id': best_match['id'],
                'concept_name': best_match['name']
            })
            print(f"Processed '{raw_event_input}': Best match is ID {best_match['id']} ('{best_match['name']}').")  # noqa: E501

    # Export to CSV
    if results_for_csv:
        df = pd.DataFrame(results_for_csv)
        df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete. Results saved to '{output_csv_path}'.")
    else:
        print("\nNo results to save.")


if __name__ == '__main__':
    input_json_file = 'similar_results.json'
    output_csv_file = 'matches.csv'

    process_json_and_export_csv(input_json_file, output_csv_file, limit=10000)
