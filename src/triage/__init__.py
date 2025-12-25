"""
Triage Module
=============

Bounded Context for ticket classification and RAG-based response generation.

Responsibilities:
- Classify tickets by product area (CASB, SWG, ZTNA, etc.) and urgency
- Generate responses based on documentation using RAG
- Provide citations to source documents

Functional Requirements Implemented:
- FR-1: POST /classify, POST /respond endpoints
- FR-2: Document ingestion into vector database
- FR-3: RAG pipeline for response generation
- FR-4: Store classification results in PostgreSQL
"""

__version__ = "1.0.0"
