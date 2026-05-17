# HR Candidate Screening Assistant

Multi-Agent RAG chatbot for searching, analyzing, and ranking candidates from a resume dataset.

## Project Architecture

```text
User Question
↓
Query Understanding Agent
↓
Resume Search Agent / RAG Tool
↓
Resume Analyst Agent
↓
Candidate Ranking Agent
↓
Final Response Agent
↓
Streamlit UI
```

## Files

```text
main.py                 Streamlit app
create_vector_db.py     Reads Resume.csv, creates embeddings, uploads to Qdrant
rag_tools.py            RAG tool for searching resume chunks from Qdrant
agents.py               Multi-agent workflow using LangGraph
prompts.py              Prompt templates for each agent
config.py               Config/secrets loader
requirements.txt        Python dependencies
.env.example            Local environment template
.streamlit/secrets.toml.example  Streamlit secrets template
data/Resume.csv         Dataset location
```

## 1. Install Libraries

```bash
pip install -r requirements.txt
```

## 2. Prepare Secrets Locally

Copy `.env.example` into `.env`:

```bash
cp .env.example .env
```

Then fill these values:

```text
OPENAI_API_KEY="..."
QDRANT_URL="..."
QDRANT_API_KEY="..."
```

## 3. Put Dataset

Put your dataset here:

```text
data/Resume.csv
```

Required columns:

```text
ID
Resume_str
Category
```

## 4. Create Vector Database

Run once before using the Streamlit app:

```bash
python create_vector_db.py
```

This will:

```text
Resume.csv
↓
clean text
↓
chunk resume text
↓
OpenAI embedding
↓
upload vectors to Qdrant Cloud
```

## 5. Run Streamlit

```bash
streamlit run main.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run main.py
```

## 6. Streamlit Cloud Deployment

On Streamlit Cloud, add secrets in App Settings > Secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key"
QDRANT_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION_NAME = "resume_collection"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
TOP_K = "5"
IDR_PER_USD = "17000"
```

Do not upload real `.env` or real `.streamlit/secrets.toml` to GitHub.

## Example Questions

```text
Cari kandidat yang punya pengalaman Python dan SQL.
```

```text
Kandidat mana yang cocok untuk role Data Analyst?
```

```text
Cari kandidat dengan pengalaman sales dan business development.
```

```text
Apa kandidat terbaik untuk posisi HR Manager?
```

## Multi-Agent Explanation

This project has five agent nodes:

1. `query_understanding_agent`  
   Converts user question into a focused semantic search query.

2. `resume_search_agent`  
   Calls the RAG tool to search relevant resume chunks from Qdrant.

3. `resume_analyst_agent`  
   Analyzes retrieved resume evidence.

4. `candidate_ranking_agent`  
   Ranks candidates based on evidence.

5. `final_response_agent`  
   Writes the final answer for the user.

The RAG tool is in `rag_tools.py`:

```python
@tool
def search_resume_tool(query: str, k: int = TOP_K):
    ...
```

The workflow is in `agents.py`:

```python
workflow.add_node("query_understanding_agent", query_understanding_agent)
workflow.add_node("resume_search_agent", resume_search_agent)
workflow.add_node("resume_analyst_agent", resume_analyst_agent)
workflow.add_node("candidate_ranking_agent", candidate_ranking_agent)
workflow.add_node("final_response_agent", final_response_agent)
```
