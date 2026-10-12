# AMD AI League Match 3 — Retrieval-Augmented Generation (RAG)

An enterprise-grade, fully dynamic Retrieval-Augmented Generation (RAG) system engineered for the AMD AI League Match 3 challenge on AMD ROCm Instinct GPU infrastructure.

---

## 1. Challenge Architecture & Specifications

This container satisfies all formal evaluation gates:

- **Startup Indexing Pass**: `python3 /app/app.py --index /app/corpus`
  - Traverses the entire corpus, parses all supported file formats, builds dense semantic + lexical indexes, and signals the persistent background daemon.
  - Startup budget: **<= 10 minutes** (including container boot and model load).
- **Per-Question Query Invocation**: `python3 /app/app.py --corpus /app/corpus --query-id <query_id> --query "<question>"`
  - High-speed thin client connecting to the memory-resident model server over a UNIX domain socket (`/app/.mc3_server.sock`).
  - Per-question latency budget: **<= 30 seconds**.
- **Standardized Output**: `/app/output/<query_id>_output.json`
  ```json
  {
    "answer": "94",
    "citations": [
      "specs/tq40_datasheet_r2.pdf"
    ],
    "confidence": 0.95
  }
  ```
- **Evaluation Criteria**:
  - **Answer**: Exact normalized value only (numbers, part codes, quarters with fiscal year, revisions with prefixes).
  - **Citations**: Exact set of necessary source files (**Necessity Test**: a file is cited if and only if removing it would make the answer impossible).
  - **Refusal**: Returns `{"answer": "", "citations": [], "confidence": 0.0}` when the answer is not present in readable corpus documents.

---

## 2. Dynamic Pipeline Design Across All 10 Benchmark Parameters

Unlike static solutions that rely on hardcoded query patterns, this system implements a completely generalized pipeline:

1. **Multi-Format Resilient Ingestion**:
   - `.pdf`: Multi-page PyPDF extraction with graceful encryption detection (`FileNotDecryptedError` caught and skipped).
   - `.docx`: Structured paragraph and multi-column table extraction.
   - `.xlsx` & `.csv`: Column-header-associated tabular representation (`header: val | header: val`), enabling exact row-level attribute search.
   - `.py`: AST-aware code analysis extracting constants, configuration defaults, and unit-expanded identifier descriptions.
   - `.png` & `.jpg`: Multi-pass Tesseract OCR (PSM 3 standard page layout + PSM 11 sparse text) to extract diagram labels, pinout numbers, and asset tags.
   - **Corpus Robustness**: Protected directory walk (`os.walk` with `onerror` handler) that safely bypasses empty folders (`archive/`), `chmod 000` unreadable files, encrypted documents, and unknown file extensions (`.dat`).

2. **Hybrid Semantic + Lexical Retrieval**:
   - Dense embeddings via `bge-small-en-v1.5` normalized for cosine similarity.
   - Token-level lexical matching with boosted weights for technical identifiers (part codes, error codes, ticket numbers).
   - Multi-hop query expansion that links production log incidents (e.g. thermal throttle alerts) to authoritative bug database resolutions.

3. **LLM Generation & Qualifier Enforcement**:
   - `Qwen2.5-0.5B-Instruct` operating in strict ChatML mode with value-only extraction.
   - Automatic preservation of complete qualifiers (e.g., fiscal year in quarters `Q3 FY27`, board revision prefixes `REV-C2`).
   - Active revision prioritization over superseded/withdrawn documents.

4. **Necessity-Based Citation Engine**:
   - Filters candidate files by verifiable presence of the extracted answer.
   - Enforces multi-hop chain citation (citing both the incident source and the resolution source).
   - Strips incidental mentions and background documents.

---

## 3. Benchmark Verification Results (10/10 PASS)

| Query ID | Question | Generated Answer | Citations | Status |
| :--- | :--- | :--- | :--- | :--- |
| `query_01` | Maximum junction temperature of TQ-40? | `94` | `specs/tq40_datasheet_r2.pdf` | **PASS** |
| `query_02` | Quarter TQ-60 enters customer sampling? | `Q3 FY27` | `planning/roadmap_fy27.docx` | **PASS** |
| `query_03` | Part number of field-replaceable fan assembly? | `ORR-FAN-2214-B` | `support/rma_parts.xlsx` | **PASS** |
| `query_04` | Firmware version that fixed ticket ORR-1847? | `4.3.2` | `support/bug_database.csv` | **PASS** |
| `query_05` | Error code logged when thermal throttle engages? | `E7731` | `logs/prod_inference_2026-09-02.log` | **PASS** |
| `query_06` | Default batch timeout, in seconds, in ingest service? | `180` | `engineering/ingest_service.py` | **PASS** |
| `query_07` | Backplane pin that carries THERM_ALERT#? | `B14` | `specs/backplane_pinout.png` | **PASS** |
| `query_08` | Board revision printed on the asset label? | `REV-C2` | `support/asset_label.jpg` | **PASS** |
| `query_09` | Firmware release fixing production log thermal incident? | `4.3.2` | `logs/prod_inference_2026-09-02.log`, `support/bug_database.csv` | **PASS** |
| `query_10` | Unit price of TQ-40 at 10,000 volume? | `""` | `[]` | **PASS** |

---

## 4. Container Packaging & Submission Guide (GHCR)

### Step 1: Log In to GitHub Container Registry
Generate a GitHub Personal Access Token (Classic) with `write:packages`, `read:packages`, and `delete:packages` permissions.
```bash
echo $CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

### Step 2: Build the Docker Image
```bash
docker build -t ghcr.io/YOUR_GITHUB_USERNAME/amd-retrieval-augmented-generation:latest .
```

### Step 3: Verify the Image Locally
```bash
# Verify ROCm PyTorch integrity
docker run --rm ghcr.io/YOUR_GITHUB_USERNAME/amd-retrieval-augmented-generation:latest \
    python3 -c "import torch; print('PyTorch build:', torch.__version__)"
# Output must end in +rocm...

# Run the container in background
docker run -d --name mc3-test ghcr.io/YOUR_GITHUB_USERNAME/amd-retrieval-augmented-generation:latest

# Run indexing pass
docker exec mc3-test python3 /app/app.py --index /app/corpus

# Run test query
docker exec mc3-test python3 /app/app.py --corpus /app/corpus --query-id q1 --query "What is the maximum junction temperature of the TQ-40?"

# Inspect output
docker exec mc3-test cat /app/output/q1_output.json
docker stop mc3-test && docker rm mc3-test
```

### Step 4: Push Image to GitHub Container Registry
```bash
docker push ghcr.io/YOUR_GITHUB_USERNAME/amd-retrieval-augmented-generation:latest
```

### Step 5: Make the Package Public on GitHub
1. Go to your GitHub profile -> **Packages** -> Select `amd-retrieval-augmented-generation`.
2. Under **Package settings**, scroll down to **Danger Zone** -> **Change package visibility** -> Select **Public**.
3. Copy the package URL (e.g., `ghcr.io/YOUR_GITHUB_USERNAME/amd-retrieval-augmented-generation:latest`) into the AMD AI League submission portal.
