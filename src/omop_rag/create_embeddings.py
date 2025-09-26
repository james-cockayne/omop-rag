import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os


def vectorise_clinical_concepts(csv_file_path, output_dir, batch_size=32):
    """
    Vectorises a CSV of clinical concepts using the MedEmbed-Large-v1 model.

    Args:
        csv_file_path (str): The path to the input CSV file. The CSV must
                             have a column named 'concept_name' containing
                             the text to embed.
        output_dir (str): The directory to save the output files.
        batch_size (int): The number of concepts to process in each batch.
    """

    print("Loading the MedEmbed-Large-v1 model...")
    model = SentenceTransformer('abhinand/MedEmbed-large-v0.1')

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded and running on: {device}")

    # Load the clinical concepts from the CSV
    try:
        df = pd.read_csv(csv_file_path)
        if 'concept_name' not in df.columns:
            raise ValueError("file must contain a column named 'concept_name'")
        concepts = df['concept_name'].tolist()
        print(f"Loaded {len(concepts)} concepts from the CSV.")
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Process concepts in batches
    embeddings = []
    print("Starting vectorisation...")
    for i in range(0, len(concepts), batch_size):
        batch = concepts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_tensor=True,
            device=device
        )
        embeddings.append(batch_embeddings.cpu())
        print(f"Processed batch {i // batch_size + 1}/{len(concepts) // batch_size + 1}")  # noqa: E501

    final_embeddings = torch.cat(embeddings)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings to a file
    embeddings_file_path = os.path.join(
        output_dir,
        'concept_embeddings.pt'
    )
    torch.save(final_embeddings, embeddings_file_path)
    print(f"Embeddings saved to {embeddings_file_path}")

    concepts_file_path = csv_input_file
    df.to_csv(concepts_file_path, index=False)
    print(f"Original concepts saved to {concepts_file_path}")

    print("Vectorisation complete!")
    return final_embeddings


if __name__ == '__main__':
    csv_input_file = 'lab_concepts.csv'
    output_directory = 'lab_embeddings'

    # Run script
    embeddings = vectorise_clinical_concepts(
        csv_input_file,
        output_directory,
        batch_size=32
    )

    if embeddings is not None:
        print(f"\nExample of the first embedding's shape: {embeddings[0].shape}")  # noqa: E501
        print(f"Example of the first embedding vector:\n{embeddings[0][:5]}...")  # noqa: E501
