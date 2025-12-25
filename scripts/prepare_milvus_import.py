#!/usr/bin/env python3
"""
Prepare Milvus Import Data
==========================

Generates embeddings for documents and creates JSON data for Milvus API insertion.
"""

import json
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
import asyncio


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for texts using OpenAI-compatible API (Groq or OpenAI)."""
    from src.config import settings

    # Use OpenAI API (works with Groq, OpenAI, or any OpenAI-compatible provider)
    # For embeddings, we need actual embeddings model - OpenAI's text-embedding-3-small
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY required for embeddings generation")

    client = AsyncOpenAI(api_key=api_key)

    embeddings = []
    for text in texts:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    return embeddings


async def main():
    """Main function to prepare Milvus import data."""

    # Load documents
    docs_file = Path(__file__).parent.parent / "docs" / "milvus_import.json"
    with open(docs_file) as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")

    # Generate embeddings
    print("Generating embeddings...")
    texts = [doc["text"] for doc in documents]
    embeddings = await generate_embeddings(texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Prepare Milvus data format
    milvus_data = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        milvus_data.append({
            "primary_key": i + 1,  # Auto-increment style
            "vector": embedding,
            "text": doc["text"],
            "doc_id": doc["id"],
            "title": doc["metadata"]["title"],
            "category": doc["metadata"]["category"],
            "url": doc["metadata"]["url"],
            "tags": json.dumps(doc["metadata"]["tags"]),
            "updated_at": doc["metadata"]["updated_at"]
        })

    # Save to file
    output_file = Path(__file__).parent.parent / "docs" / "milvus_api_import.json"
    with open(output_file, "w") as f:
        json.dump(milvus_data, f, indent=2)

    print(f"\nSaved Milvus API import data to: {output_file}")

    # Generate individual curl commands for each document
    curl_file = Path(__file__).parent.parent / "docs" / "milvus_import_curls.sh"
    with open(curl_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Milvus API Import Commands\n")
        f.write("# Run these commands to import documents one by one\n\n")

        for data in milvus_data:
            # Simplify the curl command with just the first doc
            doc_id = data["doc_id"]
            json_data = json.dumps([data], separators=(",", ":"))
            f.write(f"# Import {doc_id}\n")
            f.write(f'curl --request POST \\\n')
            f.write(f'  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/insert" \\\n')
            f.write(f'  --header "Accept: application/json" \\\n')
            f.write(f'  --header "Authorization: Bearer $MILVIZ_API_KEY" \\\n')
            f.write(f'  --header "Content-Type: application/json" \\\n')
            f.write(f'  --data \'{{"collectionName":"tickets_docs","data":{json_data}}}\'\n\n')

    print(f"Saved curl commands to: {curl_file}")

    # Print a sample
    print("\n" + "="*60)
    print("SAMPLE DATA STRUCTURE:")
    print("="*60)
    print(json.dumps(milvus_data[0], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
