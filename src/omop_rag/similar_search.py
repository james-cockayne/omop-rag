import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os
import json


def query_vector_db_to_json(
    model,
    query_text,
    concepts_file,
    embeddings_file,
    top_k=10
):
    """
    Queries a vector database of clinical concepts and returns the top_k
    results as a JSON object.

    Args:
        model (SentenceTransformer): The pre-initialised model.
        query_text (str): The clinical concept to query for.
        concepts_file (str): The path to the file with the original concepts.
        embeddings_file (str): Path to the PyTorch file with the embeddings.
        top_k (int): The number of most similar concepts to retrieve.

    Returns:
        dict: A dictionary representing the JSON output.
    """

    try:
        df = pd.read_csv(concepts_file)
        concepts = df.to_dict('records')
        stored_embeddings = torch.load(embeddings_file)
        print(f"Searching {len(concepts)} concepts and their embeddings.")
    except FileNotFoundError as e:
        print(f"Error: Required file was not found. Please check paths. {e}")
        return None
    except KeyError as e:
        print(f"Error: The expected column was not found. Please ensure the CSV file has a 'concept_name' and 'concept_id' column. {e}")  # noqa: E501
        return None

    # Vectorise the input query using the passed model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
        device=device
    )

    # Calculate cosine similarity and find top-k matches
    cosine_scores = util.cos_sim(query_embedding, stored_embeddings)[0]
    top_k_indices = torch.topk(cosine_scores, k=top_k, largest=True).indices

    # Build the JSON output
    similar_concepts = []
    for i in top_k_indices:
        score = cosine_scores[i].item()
        concept = concepts[i]
        similar_concepts.append({
            "id": concept['concept_id'],
            "name": concept['concept_name'],
            "score": round(score, 4)
        })

    return {
        "input": query_text,
        "similar_concepts": similar_concepts
    }


def process_queries_from_csv(
    model,
    input_csv_path,
    output_json_path,
    concepts_file,
    embeddings_file,
    top_k=10
):
    """
    Reads the CSV file, queries the vector database for each
    row, and writes the results to a JSON file.

    Args:
        model (SentenceTransformer): The pre-initialised model.
        input_csv_path (str): Path to the input CSV file with queries.
        output_json_path (str): Path to the output JSON file.
        concepts_file (str): The path to the file with the original concepts.
        embeddings_file (str): Path to the PyTorch file with the embeddings.
        top_k (int): The number of most similar concepts to retrieve.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The input CSV file '{input_csv_path}' was not found.")
        return

    all_results = []
    for query_text in df['EVENT']:
        # Pass the model object to the query function
        result = query_vector_db_to_json(
            model,
            query_text,
            concepts_file,
            embeddings_file,
            top_k
        )
        if result:
            all_results.append(result)

    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nProcessing complete. Results saved to '{output_json_path}'.")


if __name__ == '__main__':
    output_directory = 'med_embeddings_output'

    concepts_file_path = os.path.join(
        output_directory,
        'clinical_concepts.csv'
    )
    embeddings_file_path = os.path.join(
        output_directory,
        'clinical_concepts_embeddings.pt'
    )

    input_csv_file = 'events.csv'
    output_json_file = 'similar_results.json'

    print("Initialising the MedEmbed-Large-v1 model...")
    model = SentenceTransformer('abhinand/MedEmbed-large-v0.1')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model initialised and running on: {device}")

    # Run the main processing function
    process_queries_from_csv(
        model,
        input_csv_file,
        output_json_file,
        concepts_file_path,
        embeddings_file_path,
        top_k=10
    )
