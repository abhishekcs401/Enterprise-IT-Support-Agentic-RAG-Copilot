# from pathlib import Path

# from app.services.ingestion import chunk_documents, load_file

# from app.rag.vectorstore import add_documents
# docs = load_file(Path("data/sample_kb/company_it_handbook.md"))

# chunks = chunk_documents(docs)
# add_documents(chunks)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)



