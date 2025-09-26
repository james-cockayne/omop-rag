import os
import argparse
from src.omop_rag.create_embeddings import vectorise_clinical_concepts
from src.omop_rag.similar_search import process_queries_from_csv
from src.omop_rag.best_match import process_json_and_export_csv
from sentence_transformers import SentenceTransformer
import torch


def main():
    """
    Main CLI
    """
    parser = argparse.ArgumentParser(
        description="Run various tasks for the project."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for create-embeddings
    create_parser = subparsers.add_parser(
        "create-embeddings",
        help="Create and save embeddings for a dataset."
    )
    create_parser.add_argument(
        '--concepts-file',
        type=str,
        required=True,
        help="Path to the concepts CSV file."
    )
    create_parser.add_argument(
        '--embeddings-file',
        type=str,
        required=True,
        help="Path to the embeddings .pt file to save."
    )

    # Subparser for similar-search
    search_parser = subparsers.add_parser(
        "similar-search",
        help="Query a vector database and find similar concepts."
    )
    search_parser.add_argument(
        '--concepts-file',
        type=str,
        required=True,
        help="Path to the concepts CSV file."
    )
    search_parser.add_argument(
        '--embeddings-file',
        type=str,
        required=True,
        help="Path to the embeddings .pt file to use."
    )
    search_parser.add_argument(
        '--input-csv',
        type=str,
        required=True,
        help="Path to the input CSV file containing queries."
    )
    search_parser.add_argument(
        '--output-json',
        type=str,
        required=True,
        help="Path to the output JSON file for results."
    )

    # Subparser for best-match
    best_match_parser = subparsers.add_parser(
        "best-match",
        help="Use a QA model to find the best match from similar concepts."
    )
    best_match_parser.add_argument(
        '--input-json',
        type=str,
        required=True,
        help="Path to the input JSON file from the similar search."
    )
    best_match_parser.add_argument(
        '--output-csv',
        type=str,
        required=True,
        help="Path to the output CSV file for the final matches."
    )

    args = parser.parse_args()

    if args.command in ['create-embeddings', 'similar-search']:
        print("Initialising the MedEmbed-Large-v1 model...")
        model = SentenceTransformer('abhinand/MedEmbed-large-v0.1')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model initialised and running on: {device}")

    if args.command == 'best-match':
        print("Model loaded.")

    if args.command == 'create-embeddings':
        print("\nStarting the embedding creation process...")

        embeddings = vectorise_clinical_concepts(
            args.concepts_file,
            os.path.dirname(args.concepts_file),
            batch_size=32
        )

        if embeddings is not None:
            print(f"\nVectorisation completed for {args.concepts_file}")
            print(f"Number of embeddings created: {embeddings.shape[0]}")
            print(f"Dimension of each embedding: {embeddings.shape[1]}")

    elif args.command == 'similar-search':
        print("\nStarting the similar concept search...")

        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

        process_queries_from_csv(
            model,
            args.input_csv,
            args.output_json,
            args.concepts_file,
            args.embeddings_file,
            top_k=10
        )

    elif args.command == 'best-match':
        print("\nStarting the best match selection process...")

        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        process_json_and_export_csv(
            args.input_json,
            args.output_csv,
            limit=10000
        )
    else:
        print("No task flag provided. Use --help to see available options.")


if __name__ == '__main__':
    main()
