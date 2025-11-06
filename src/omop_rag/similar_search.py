import json
import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


def process_queries_from_csv(
    model: SentenceTransformer,
    input_csv_path: str,
    output_json_path: str,
    concepts_file: str,
    embeddings_file: str,
    top_k: int = 10,
    batch_size: int = 128,
    processed_column: str | None = None,
):
    """Reads queries from a CSV, finds top matches, and saves to JSON.

    This function executes the entire batch processing workflow.

    Args:
        model (SentenceTransformer): The initialised sentence transformer model.
        input_csv_path (str): The file path for the input CSV. Must contain
            a column named 'EVENT' with the terms to query.
        output_json_path (str): The file path where the output JSON will be
            saved.
        concepts_file (str): The path to the CSV file containing concept
            metadata (e.g., 'concept_id', 'concept_name').
        embeddings_file (str): The path to the .pt file containing the
            pre-computed torch tensor of concept embeddings.
        top_k (int): The number of most similar concepts to retrieve for
            each query.
        batch_size (int): The batch size to use for encoding queries, which
            helps manage memory usage on the GPU/CPU.
        processed_column (str | None): Optional column to use for query
            embedding (e.g., 'EVENT_PROCESSED'). When provided, this column is
            encoded while the raw 'EVENT' text is retained in the output JSON
            under the 'input' key and the processed text is written to the
            'processed_input' key.
    """
    device = model.device

    # Step 1: Load the vector database assets into memory.
    print("Loading the vector database into memory...")
    try:
        concepts_df = pd.read_csv(concepts_file)
        concepts = concepts_df.to_dict("records")
        stored_embeddings = torch.load(
            embeddings_file, map_location=device
        ).to(torch.float16)
    except FileNotFoundError as e:
        print(f"Error: A required database file was not found. {e}")
        return

    # Step 2: Load and encode all queries from the input CSV file.
    try:
        queries_df = pd.read_csv(input_csv_path)
        if "EVENT" not in queries_df.columns:
            print("Error: Input CSV must contain an 'EVENT' column.")
            return

        raw_queries = queries_df["EVENT"].astype(str).tolist()

        if processed_column:
            if processed_column not in queries_df.columns:
                print(
                    f"Error: Input CSV must contain a '{processed_column}' column when"
                    " the --preprocessed flag is used."
                )
                return
            search_queries = queries_df[processed_column].astype(str).tolist()
            print(
                f"Found {len(search_queries)} events to process. Using '{processed_column}'"
                " for embeddings while preserving raw 'EVENT' values."
            )
        else:
            search_queries = raw_queries
            print(f"Found {len(raw_queries)} events to process. Encoding all...")
    except FileNotFoundError:
        print(f"Error: Input CSV '{input_csv_path}' not found.")
        return

    query_embeddings = model.encode(
        search_queries,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
        batch_size=batch_size,
    ).to(torch.float16)

    # Step 3: Perform a single, massive similarity search.
    print("\nPerforming similarity search for all queries...")
    cosine_scores = util.cos_sim(query_embeddings, stored_embeddings)

    # Step 4: Retrieve the top K results for each query from the score matrix.
    top_results = torch.topk(cosine_scores, k=top_k, dim=1)

    # Step 5: Format the results and prepare for JSON export.
    print("Formatting results...")
    all_results = []
    for i, (raw_query, processed_query) in enumerate(
        zip(raw_queries, search_queries)
    ):
        scores = top_results.values[i]
        indices = top_results.indices[i]

        similar_concepts = [
            {
                "id": concepts[idx]["concept_id"],
                "name": concepts[idx]["concept_name"],
                "score": round(score.item(), 4),
            }
            for score, idx in zip(scores, indices)
        ]

        result_item: dict[str, object] = {"input": raw_query}

        if processed_column:
            result_item["processed_input"] = processed_query

        result_item["similar_concepts"] = similar_concepts

        all_results.append(result_item)

    with open(output_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Processing complete. Results saved to '{output_json_path}'.")


def main():
    """Defines file paths and runs the main batch processing function."""
    output_directory = "med_embeddings_output"
    concepts_file_path = os.path.join(
        output_directory, "clinical_concepts.csv"
    )
    embeddings_file_path = os.path.join(
        output_directory, "clinical_concepts_embeddings.pt"
    )

    input_csv_file = "events.csv"
    output_json_file = "similar_results.json"

    # Initialise the sentence transformer model.
    # Using float16 can significantly speed up computation on compatible GPUs.
    print("Initialising the MedEmbed-Large-v1 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "abhinand/MedEmbed-large-v0.1",
        device=device,
        model_kwargs={"dtype": torch.float16},
    )
    print(f"Model initialised and running on: {device}")

    # Run the main processing function.
    process_queries_from_csv(
        model=model,
        input_csv_path=input_csv_file,
        output_json_path=output_json_file,
        concepts_file=concepts_file_path,
        embeddings_file=embeddings_file_path,
        top_k=10,
        batch_size=128,  # Adjust this based on your GPU's VRAM.
    )


if __name__ == "__main__":
    main()
