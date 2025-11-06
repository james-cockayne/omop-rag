import os
import time

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


def load_vector_db_assets(concepts_file, embeddings_file):
    """Load the Sentence Transformer model, concepts, and embeddings.

    This function is an expensive, one-time operation that prepares all
    necessary assets for querying.

    Args:
        concepts_file (str): The file path for the CSV containing concept
            data (e.g., 'concept_id', 'concept_name').
        embeddings_file (str): The file path for the .pt file containing
            the pre-computed torch tensor of embeddings.

    Returns:
        tuple: A tuple containing the model, a list of concept
        dictionaries, the embeddings tensor, and the device string.
        Returns (None, None, None, None) on critical model loading errors.
    """
    start_time = time.time()
    print("--- Initialising Vector DB Assets ---")

    try:
        model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded and running on: {device}")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        return None, None, None, None

    try:
        df = pd.read_csv(concepts_file)
        concepts = df.to_dict("records")
        stored_embeddings = torch.load(embeddings_file)
        # Move the loaded embeddings tensor to the same device as the model.
        stored_embeddings = stored_embeddings.to(device)
        print(
            f"Loaded {len(concepts)} concepts and their embeddings from disk."
        )
    except FileNotFoundError as e:
        print(f"Error: Asset file not found. Please check paths. {e}")
        return model, None, None, device
    except KeyError as e:
        print(f"Error: Missing required column in CSV. {e}")
        return model, None, None, device

    end_time = time.time()
    print(f"Asset loading completed in {end_time - start_time:.2f} seconds.")
    print("---------------------------------------")

    return model, concepts, stored_embeddings, device


def query_vector_db(
    model, query_text, concepts_list, embeddings_tensor, device, top_k=5
):
    """Query the vector database to find the most similar concepts.
    
    Args:
        model (SentenceTransformer): The pre-initialised sentence
            transformer model.
        query_text (str): The clinical concept or phrase to search for.
        concepts_list (list): A list of dictionaries, where each
            dictionary represents a concept.
        embeddings_tensor (torch.Tensor): The tensor containing all
            pre-computed concept embeddings.
        device (str): The device the model is running on ('cuda' or 'cpu').
        top_k (int): The number of top matching concepts to return.

    Returns:
        list: A list of dictionaries, each containing the 'id', 'name',
        and 'score' of a matching concept. Returns an empty list if
        assets are not loaded correctly.
    """
    start_time = time.time()

    if model is None or not concepts_list or embeddings_tensor is None:
        print("Error: Vector DB assets are not loaded. Cannot run query.")
        return []

    query_embedding = model.encode(
        query_text, convert_to_tensor=True, device=device
    )

    embeddings_tensor = embeddings_tensor.to(query_embedding.dtype)

    cosine_scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
    top_k_results = torch.topk(cosine_scores, k=top_k)

    results = []
    print(f"\nQuery: '{query_text}'")
    print(f"Top {top_k} most similar clinical concepts found:")

    for score, idx in zip(top_k_results.values, top_k_results.indices):
        concept = concepts_list[idx]
        score = score.item()
        print(
            f"  - ID: {concept['concept_id']}, "
            f"Name: '{concept['concept_name']}' (Score: {score:.4f})"
        )
        results.append(
            {
                "id": concept["concept_id"],
                "name": concept["concept_name"],
                "score": score,
            }
        )

    end_time = time.time()
    print(f"Query executed in {end_time - start_time:.4f} seconds.")

    return results


if __name__ == "__main__":
    output_directory = "med_embeddings_output"
    concepts_file_path = os.path.join(
        output_directory, "clinical_concepts.csv"
    )
    embeddings_file_path = os.path.join(
        output_directory, "clinical_concepts_embeddings.pt"
    )

    MODEL, CONCEPTS, EMBEDDINGS, DEVICE = load_vector_db_assets(
        concepts_file_path, embeddings_file_path
    )

    if EMBEDDINGS is not None:
        results1 = query_vector_db(
            model=MODEL,
            query_text="German turnip",
            concepts_list=CONCEPTS,
            embeddings_tensor=EMBEDDINGS,
            device=DEVICE,
            top_k=5,
        )

        results2 = query_vector_db(
            model=MODEL,
            query_text="serum glucose high",
            concepts_list=CONCEPTS,
            embeddings_tensor=EMBEDDINGS,
            device=DEVICE,
            top_k=3,
        )
