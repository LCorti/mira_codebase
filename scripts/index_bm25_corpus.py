#!/usr/bin/env python3
"""
Script to index a corpus from HuggingFace datasets using BM25s and save it for later use.

Example usage:
    python index_bm25_corpus.py \
        --dataset-name "BeIR/scifact" \
        --split "corpus" \
        --text-column "text" \
        --id-column "_id" \
        --output-dir "./bm25_indices/scifact" \
        --stemming-language "english" \
        --stopword-language "english"
"""

import argparse
import logging
import sys
from pathlib import Path

import bm25s
import Stemmer
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def index_corpus(
    dataset_name: str,
    split: str,
    text_column: str,
    id_column: str,
    output_dir: str,
    config: str = "",
    stemming_language: str = "",
    stopword_language: str = "",
    save_corpus: bool = True,
):
    """
    Index a corpus from HuggingFace using BM25s and save it.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use (e.g., 'train', 'test', or None for all)
        text_column: Column name containing the text to index
        id_column: Column name containing document IDs
        output_dir: Directory to save the index
        config: Optional config name for datasets with multiple configs
        stemming_language: Optional language for stemming (e.g., 'english')
        stopword_language: Optional language for stopwords (e.g., 'english')
        save_corpus: Whether to save the corpus along with the index
    """
    logger.info(
        f"Loading dataset '{dataset_name}' (config: '{config}', split: '{split}')..."
    )

    try:
        # Load dataset directly from HuggingFace
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)

        # Extract corpus and IDs
        corpus = list(dataset[text_column])
        corpus_ids = [str(doc_id) for doc_id in dataset[id_column]]

        logger.info(f"Loaded {len(corpus)} documents from '{dataset_name}'")

    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_name}': {e}")
        raise

    # Setup stemmer if specified
    stemmer = None
    if stemming_language:
        try:
            stemmer = Stemmer.Stemmer(stemming_language)
            logger.info(f"Using stemmer for language: {stemming_language}")
        except Exception as e:
            logger.warning(
                f"Could not initialize stemmer for '{stemming_language}': {e}"
            )

    # Tokenize corpus
    logger.info("Tokenizing corpus...")
    try:
        corpus_tokens = bm25s.tokenize(
            corpus, stopwords=stopword_language, stemmer=stemmer
        )
        logger.info("Tokenization complete")
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        raise

    # Create and index retriever
    logger.info("Building BM25 index...")
    try:
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        logger.info("Index built successfully")
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the index
    logger.info(f"Saving index to '{output_dir}'...")
    try:
        retriever.save(str(output_path), corpus=corpus if save_corpus else None)
        logger.info("Index saved successfully")
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise

    # Save document IDs separately
    ids_path = output_path / "doc_ids.txt"
    logger.info(f"Saving document IDs to '{ids_path}'...")
    try:
        with open(ids_path, "w") as f:
            f.writelines(f"{doc_id}\n" for doc_id in corpus_ids)
        logger.info(f"Saved {len(corpus_ids)} document IDs")
    except Exception as e:
        logger.error(f"Error saving document IDs: {e}")
        raise

    # Save metadata
    metadata_path = output_path / "metadata.txt"
    logger.info(f"Saving metadata to '{metadata_path}'...")
    try:
        with open(metadata_path, "w") as f:
            f.write(f"dataset_name: {dataset_name}\n")
            f.write(f"config: {config}\n")
            f.write(f"split: {split}\n")
            f.write(f"text_column: {text_column}\n")
            f.write(f"id_column: {id_column}\n")
            f.write(f"num_documents: {len(corpus)}\n")
            f.write(f"stemming_language: {stemming_language}\n")
            f.write(f"stopword_language: {stopword_language}\n")
        logger.info("Metadata saved successfully")
    except Exception as e:
        logger.warning(f"Error saving metadata: {e}")

    logger.info(f"\n> Successfully indexed corpus from '{dataset_name}'")
    logger.info(f"  - Documents: {len(corpus)}")
    logger.info(f"  - Output: {output_dir}")
    logger.info("\nTo load this index later:")
    logger.info(
        f"  retriever = bm25s.BM25.load('{output_dir}', load_corpus={'True' if save_corpus else 'False'})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Index a HuggingFace corpus using BM25s and save it for later use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index the SciFact corpus
    python index_bm25_corpus.py \\
        --dataset-name "BeIR/scifact" \\
        --config "corpus" \\
        --text-column "text" \\
        --id-column "_id" \\
        --output-dir "./bm25_indices/scifact"

    # Index TREC-COVID corpus with stemming and stopwords
    python index_bm25_corpus.py \\
        --dataset-name "BeIR/trec-covid" \\
        --config "corpus" \\
        --text-column "text" \\
        --id-column "_id" \\
        --output-dir "./bm25_indices/trec_covid" \\
        --stemming-language "english" \\
        --stopword-language "english"
        """,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help='Name of the HuggingFace dataset (e.g., "BeIR/scifact")',
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help='Config name for datasets with multiple configs (e.g., "corpus", "queries")',
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help='Dataset split to use (e.g., "train", "test"). Use None for all data.',
    )

    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Column name containing the text to index",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default="_id",
        help='Column name containing document IDs (default: "_id")',
    )

    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save the BM25 index"
    )

    parser.add_argument(
        "--stemming-language",
        type=str,
        default=None,
        help='Language for stemming (e.g., "english", "french"). Optional.',
    )

    parser.add_argument(
        "--stopword-language",
        type=str,
        default=None,
        help='Language for stopwords (e.g., "english", "french"). Optional.',
    )

    parser.add_argument(
        "--no-save-corpus",
        action="store_true",
        help="Do not save the corpus text (saves disk space, but requires dataset for retrieval)",
    )

    args = parser.parse_args()

    try:
        index_corpus(
            dataset_name=args.dataset_name,
            split=args.split,
            text_column=args.text_column,
            id_column=args.id_column,
            output_dir=args.output_dir,
            config=args.config,
            stemming_language=args.stemming_language,
            stopword_language=args.stopword_language,
            save_corpus=not args.no_save_corpus,
        )
    except Exception as e:
        logger.error(f"Failed to index corpus: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
