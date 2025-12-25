#!/usr/bin/env python3
"""
Generate Milvus Import with Embeddings
======================================

Generate embeddings for documents using sentence-transformers (free local model).
Creates JSON file ready for Milvus/Zilliz REST API import.
"""

import json
import sys
from pathlib import Path


def generate_embeddings():
    """Generate embeddings using sentence-transformers."""

    # Check if sentence-transformers is installed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Load documents
    docs_file = Path(__file__).parent.parent / "docs" / "milvus_ui_import.json"
    if not docs_file.exists():
        print(f"Error: File not found: {docs_file}")
        sys.exit(1)

    with open(docs_file) as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents from {docs_file}")

    # Load model (768 dimensions, good balance of speed/quality)
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings
    print("Generating embeddings...")
    data = []
    for i, doc in enumerate(docs):
        # Encode text to get vector
        vector = model.encode(doc['text'], show_progress_bar=False).tolist()

        data.append({
            "primary_key": doc["primary_key"],
            "vector": vector,
            "text": doc['text'],
            "title": doc['title'],
            "category": doc['category'],
            "url": doc['url'],
            "tags": doc['tags'],
            "updated_at": doc['updated_at']
        })

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(docs)} documents...")

    print(f"Generated {len(data)} documents with embeddings")

    # Save to file
    output_file = Path(__file__).parent.parent / "docs" / "milvus_with_embeddings.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print sample
    print("\n" + "="*60)
    print("SAMPLE DATA (first record):")
    print("="*60)
    sample = data[0].copy()
    sample['vector'] = sample['vector'][:5] + [f"... ({len(sample['vector'])} total)"]
    print(json.dumps(sample, indent=2))

    # Print import command
    print("\n" + "="*60)
    print("IMPORT COMMAND:")
    print("="*60)
    print("""Set your credentials:
export MILVUS_ENDPOINT="https://in03-2f74f35669a52a3.serverless.gcp-us-west1.cloud.zilliz.com"
export MILVIZ_API_KEY="your-api-key"

Import data:
curl --request POST \\
  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/insert" \\
  --header "Authorization: Bearer $MILVIZ_API_KEY" \\
  --header "Content-Type: application/json" \\
  -d @docs/milvus_with_embeddings.json""")


if __name__ == "__main__":
    generate_embeddings()
