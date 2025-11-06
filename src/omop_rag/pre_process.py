import json
import logging
import re
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

_DEFAULT_ACRONYM_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "lab" / "acronyms.json"
)


def load_acronym_map(json_path: Path | str = _DEFAULT_ACRONYM_PATH) -> dict[str, str]:
    path = Path(json_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        logger.error(
            "Acronym map file not found at '%s'. Falling back to empty map.",
            path,
        )
        return {}
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse acronym map JSON at '%s': %s",
            path,
            exc,
        )
        return {}

    if not isinstance(data, dict):
        logger.error(
            "Acronym map JSON at '%s' is not an object. Falling back to empty map.",
            path,
        )
        return {}

    logger.info("Loaded %d acronym expansions from %s", len(data), path)
    return {str(key): str(value) for key, value in data.items()}


ACRONYM_MAP = load_acronym_map()


_REGEX_NORMALISATIONS = [
    (
        re.compile(r"\bcopy\s*no\.?\b", re.IGNORECASE), "copy number"
    ),
    (
        re.compile(r"\bcarrier sequence\b", re.IGNORECASE),
        "carrier gene sequencing",
    ),
    (
        re.compile(r"\bpattern level\b", re.IGNORECASE), "pattern titre"
    ),
    (
        re.compile(r"\bTranscript\s*/\s*ABL\b", re.IGNORECASE),
        "ABL transcript ratio",
    ),
    (
        re.compile(r"\bPneumoccocal\b", re.IGNORECASE), "Pneumococcal"
    ),
    (
        re.compile(r"\bHaemglobinopathy\b", re.IGNORECASE), "Haemoglobinopathy"
    ),
    (
        re.compile(r"\bFluores\.\b", re.IGNORECASE), "Fluorescence"
    ),
    (
        re.compile(r"\s*\(BG\)\b", re.IGNORECASE), " blood gas"
    ),
    (
        re.compile(r"\bOther (haematology|immunology) test special result\b", re.IGNORECASE),
        r"Other \1 test result",
    ),
]


def _normalise_event_text(text: str) -> str:
    """Applies project-specific text normalisations prior to acronym expansion."""

    normalised = text
    for pattern, replacement in _REGEX_NORMALISATIONS:
        normalised = pattern.sub(replacement, normalised)

    normalised = re.sub(r"\s{2,}", " ", normalised)
    return normalised.strip()


def _clean_and_expand_event(text: str) -> tuple[str, int]:
    """Removes prefixes and expands acronyms in a single text string.

    This helper function first removes a specific "O_" prefix and then
    iterates through a predefined map of acronyms, replacing them with
    their expanded forms case-insensitively.

    Args:
        text (str): The input string to process.

    Returns:
        tuple[str, int]: The processed text string and the total number
        of acronym matches expanded.
    """
    processed_text = re.sub(r"^O_", "", text)
    total_replacements = 0

    # Iterate through the global map and expand all known acronyms. The use of
    # word boundaries (\b) prevents partial matches (e.g., matching 'Fe' in
    # 'Female'). We use re.subn to capture how many replacements occurred for
    # logging and summary statistics.
    for acronym, expansion in ACRONYM_MAP.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        processed_text, replacements = re.subn(
            pattern,
            expansion,
            processed_text,
            flags=re.IGNORECASE,
        )
        if replacements:
            logger.debug(
                "Expanded acronym '%s' to '%s' (%d occurrence%s) in text '%s'",
                acronym,
                expansion,
                replacements,
                "s" if replacements != 1 else "",
                text,
            )
            total_replacements += replacements

    return processed_text, total_replacements


def process_lab_events(
    input_filepath: str = "data/lab/lab_events.csv",
    output_filepath: str = "data/lab/lab_events_processed.csv",
):
    """Reads, processes, and saves lab event data from a CSV file.

    This function orchestrates the file reading, applies the text cleaning
    and expansion logic to each relevant row, and saves the results to a
    new CSV file.

    Args:
        input_filepath (str): The path to the input CSV file. It must
            contain a column named 'EVENT'.
        output_filepath (str): The path where the processed CSV file
            will be saved. It will contain two columns: 'EVENT' and
            'EVENT_PROCESSED'.
    """
    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded '{input_filepath}'.")

        if "EVENT" not in df.columns:
            print("Error: The input file must contain an 'EVENT' column.")
            return

        df["EVENT"] = df["EVENT"].astype(str).str.strip()
        df["EVENT"] = df["EVENT"].str.replace(
            r"(?i)^POC\s*-\s*", "", regex=True
        )
        df["EVENT"] = df["EVENT"].str.replace(
            r"(?i)\s*-\s*POC$", "", regex=True
        )
        df["EVENT"] = df["EVENT"].apply(_normalise_event_text)

        mask = ~df["EVENT"].str.contains("comment", case=False, na=False)
        removed_count = int((~mask).sum())
        if removed_count:
            logger.info(
                "Removed %d comment rows prior to processing.", removed_count
            )
        df = df[mask].copy()

        processed_series = df["EVENT"].apply(
            _clean_and_expand_event
        )
        df["EVENT_PROCESSED"] = processed_series.map(lambda res: res[0])
        total_matches = processed_series.map(lambda res: res[1]).sum()

        output_df = df[["EVENT", "EVENT_PROCESSED"]]
        output_df.to_csv(output_filepath, index=False)
        print(f"Processing complete. Output saved to '{output_filepath}'.")
        logger.info("Total acronym matches expanded: %d", total_matches)

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        print(
            "Please make sure the script and the CSV file are in the same "
            "directory."
        )
    except Exception as e:  # noqa: BLE001
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_lab_events()
