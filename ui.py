import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

# Assuming the previously refactored script is at this location.
from src.omop_rag.query_data import (
    query_vector_db as single_query_vector_db,
)


@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model.

    The model is loaded onto a GPU if available, otherwise CPU.
    Streamlit's cache_resource decorator ensures this expensive operation
    runs only once per session.

    Returns:
        tuple: A tuple containing the loaded SentenceTransformer model
        and the device string ('cuda' or 'cpu'). Returns (None, None)
        on failure.
    """
    st.info(
        "Loading MedEmbed-Large-v1 (Sentence Transformer)... "
        "This may take a moment."
    )
    try:
        model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        return None, None


@st.cache_data(show_spinner=False)
def load_vector_db_data(concepts_file, embeddings_file, device): # MODIFICATION 1: Add 'device' argument
    """Load and cache the concept and embedding data from disk.

    Uses Streamlit's cache_data decorator to prevent reloading data on
    every interaction.

    Args:
        concepts_file (str): The file path for the concepts CSV.
        embeddings_file (str): The file path for the embeddings .pt tensor.
        device (str): The torch device ('cuda' or 'cpu') to move the tensor to.

    Returns:
        tuple: A status string ("Success" or an error message), a list of
        concept dictionaries, and the embeddings tensor. The latter two
        are None on failure.
    """
    try:
        concepts_df = pd.read_csv(concepts_file)
        concepts = concepts_df.to_dict("records")
        stored_embeddings = torch.load(embeddings_file)
        # MODIFICATION 2: Move the tensor to the correct device
        stored_embeddings = stored_embeddings.to(device)
        return "Success", concepts, stored_embeddings
    except FileNotFoundError as e:
        return f"Error: File not found. {e}", None, None
    except KeyError as e:
        return f"Error: Missing column in CSV. {e}", None, None
    except Exception as e:
        return f"Error: loading data: {e}", None, None


def display_query_results(results_df):
    """Render the query results in the Streamlit UI.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the query
            results with columns for ID, name, and score.
    """
    if results_df.empty:
        st.warning("Search returned no results.")
        return

    st.success(f"✅ Found {len(results_df)} closest matches.")
    st.markdown("### Top Match")
    st.dataframe(results_df.head(1), use_container_width=True)

    if len(results_df) > 1:
        st.markdown("### Other Similar Concepts")
        st.dataframe(results_df.tail(-1), use_container_width=True)


def run_single_concept_lookup():
    """Render UI elements and logic for single concept lookup."""
    st.header("Single Concept Lookup")
    st.markdown(
        "Instantly find the closest OMOP concept for a single text input."
    )

    concepts_file = st.text_input(
        "Concepts File Path (.csv)",
        "data/lab/lab_concepts.csv",
        key="single_concepts_path",
    )
    embeddings_file = st.text_input(
        "Embeddings File Path (.pt)",
        "embeddings/lab/concept_embeddings.pt",
        key="single_embeddings_path",
    )
    query_text = st.text_input(
        "Enter Concept to Query",
        value="lab test",
        help="Type the raw event name you want to map.",
    )
    top_k = st.slider(
        "Number of Top Matches to Display",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    if st.button("Find Closest Match", key="run_single_query"):
        if not query_text:
            st.error("Please enter a concept string to search.")
            return

        model, device = load_embedding_model()
        if not model:
            return

        # MODIFICATION 3: Pass the 'device' to the data loading function
        status, concepts, embeddings = load_vector_db_data(
            concepts_file, embeddings_file, device
        )
        if status != "Success":
            st.error(f"Data Loading Error: {status}")
            return

        with st.spinner(f"Searching for '{query_text}'..."):
            try:
                results = single_query_vector_db(
                    model=model,
                    query_text=query_text,
                    concepts_list=concepts,
                    embeddings_tensor=embeddings,
                    device=device,
                    top_k=top_k,
                )
                results_df = pd.DataFrame(results)
                results_df.rename(
                    columns={
                        "id": "OMOP Concept ID",
                        "name": "Concept Name",
                        "score": "Similarity Score",
                    },
                    inplace=True,
                )
                display_query_results(results_df)

            except Exception as e:
                st.error(f"Error during query execution: {e}")


def main():
    """Set up the page configuration and run the Streamlit app."""
    st.set_page_config(
        page_title="OMOP Concept Mapping RAG", layout="wide"
    )
    st.title("omop-rag")
    run_single_concept_lookup()


if __name__ == "__main__":
    main()
