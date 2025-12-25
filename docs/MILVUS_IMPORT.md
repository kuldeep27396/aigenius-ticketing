# Milvus Data Import Guide

This guide shows how to import documentation into Milvus/Zilliz Cloud for RAG-based response generation.

## Quick Start: Using sentence-transformers (Free)

**Note:** Zilliz Cloud UI import requires a `vector` field. The easiest method is to generate embeddings locally using the free `sentence-transformers` library.

### Step 1: Install Dependencies

```bash
pip install sentence-transformers
```

### Step 2: Generate Embeddings

Run the provided script:

```bash
python scripts/generate_milvus_import.py
```

This will:
- Load documents from `docs/milvus_ui_import.json`
- Generate 768-dimensional embeddings using `all-MiniLM-L6-v2`
- Save to `docs/milvus_with_embeddings.json`

### Step 3: Import via REST API

```bash
# Set your credentials
export MILVUS_ENDPOINT="https://in03-2f74f35669a52a3.serverless.gcp-us-west1.cloud.zilliz.com"
export MILVIZ_API_KEY="your-api-key"

# Import all data
curl --request POST \
  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/insert" \
  --header "Accept: application/json" \
  --header "Authorization: Bearer $MILVIZ_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{"collectionName":"tickets_docs","data":[]}'  # Empty first to create structure
```

Actually, for bulk import, it's better to use the generated file:

```bash
# Import with embeddings
curl --request POST \
  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/insert" \
  --header "Accept: application/json" \
  --header "Authorization: Bearer $MILVIZ_API_KEY" \
  --header "Content-Type: application/json" \
  -d @docs/milvus_with_embeddings.json
```

## Alternative: Manual Import via Zilliz UI

If you prefer using the Zilliz Cloud UI:

1. Go to https://cloud.zilliz.com
2. Navigate to your collection `tickets_docs`
3. Click **Insert Data**
4. Manually add records with:
   - `primary_key`: Integer (1, 2, 3, ...)
   - `vector`: [float array - 768 dimensions]
   - `text`: Document content
   - `title`, `category`, `url`, `tags`, `updated_at`

**Note:** You'll need to generate vectors separately using the script above.

## Verify Import

Check if data is loaded:

```bash
curl --request POST \
  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/query" \
  --header "Authorization: Bearer $MILVIZ_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "collectionName": "tickets_docs",
    "outputFields": ["primary_key", "title", "category"]
  }'
```

## Test RAG Response

Once data is imported, test the RAG endpoint:

```bash
curl -X POST https://aigenius-ticketing.onrender.com/triage/respond \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TEST-001",
    "subject": "CASB Salesforce sync is not working",
    "content": "Our CASB integration with Salesforce has stopped syncing data. What should I check?"
  }'
```

You should get a response with relevant documentation snippets from the imported data.
