# AMD AI League Match 3 — RAG

A Retrieval-Augmented Generation system for document question answering with source citations.

## Components
- app/app.py — indexing and query entry point
- app/server.py — retrieval and answering server
- app/requirements.txt — Python dependencies
- Dockerfile — container configuration

## Sample validation
Question: What is the maximum junction temperature of the TQ-40?

Answer: 94

Citation: specs/tq40_datasheet_r2.pdf

## Status
The sample query was tested in the AMD notebook environment.
The Docker image has not yet been built and published.
