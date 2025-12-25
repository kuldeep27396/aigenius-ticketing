# Milvus Data Import Guide

This guide shows how to import documentation into Milvus/Zilliz Cloud for RAG-based response generation.

## Method 1: Zilliz Cloud UI (Recommended)

**Steps:**

1. Go to your Zilliz Cloud cluster: https://cloud.zilliz.com

2. Navigate to your collection `tickets_docs`

3. Click **Import Data** → **JSON Import**

4. Upload `docs/milvus_ui_import.json`

5. **Enable Auto-Generated Embeddings** (Zilliz will create vectors automatically)

6. Map fields:
   - `doc_id` → Primary Key (String)
   - `text` → Vector Field (enable embedding)
   - `title` → Scalar Field
   - `category` → Scalar Field
   - `url` → Scalar Field
   - `tags` → Scalar Field (Array)
   - `updated_at` → Scalar Field

7. Click **Import**

## Method 2: REST API (Requires Embeddings)

If you want to use the API directly, you first need to generate embeddings.

### Prerequisites

```bash
# Option A: Use OpenAI embeddings (requires API key with quota)
export OPENAI_API_KEY=your-key-here

# Option B: Use local sentence-transformers (free)
pip install sentence-transformers
```

### Using sentence-transformers (Free)

```python
from sentence_transformers import SentenceTransformer
import json

# Load model (768 dimensions)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
with open('docs/milvus_ui_import.json') as f:
    docs = json.load(f)

# Generate embeddings
data = []
for i, doc in enumerate(docs):
    vector = model.encode(doc['text']).tolist()
    data.append({
        "primary_key": i + 1,
        "vector": vector,
        "text": doc['text'],
        "doc_id": doc['doc_id'],
        "title": doc['title'],
        "category": doc['category'],
        "url": doc['url'],
        "tags": doc['tags'],
        "updated_at": doc['updated_at']
    })

# Save for import
with open('docs/milvus_with_embeddings.json', 'w') as f:
    json.dump(data, f)

print(f"Generated {len(data)} documents with 768-dim embeddings")
```

### Push to Milvus via API

```bash
# Set your credentials
export MILVUS_ENDPOINT="https://in03-2f74f35669a52a3.serverless.gcp-us-west1.cloud.zilliz.com"
export MILVIZ_API_KEY="your-api-key"

# Import data
curl --request POST \
  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/insert" \
  --header "Accept: application/json" \
  --header "Authorization: Bearer $MILVIZ_API_KEY" \
  --header "Content-Type: application/json" \
  --data @docs/milvus_with_embeddings.json
```

## Verify Import

Check if data is loaded:

```bash
curl --request POST \
  --url "$MILVUS_ENDPOINT/v2/vectordb/entities/query" \
  --header "Authorization: Bearer $MILVIZ_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "collectionName": "tickets_docs",
    "outputFields": ["doc_id", "title", "category"]
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
