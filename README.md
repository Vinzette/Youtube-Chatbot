# YouTube-Chatbot

Lightweight demo that indexes a YouTube video's transcript, creates OpenAI embeddings, stores them in a FAISS vector store, and answers user questions using retrieval-augmented generation.

**What this project does**

- Uses `youtube_transcript_api` to fetch a video's captions (if available).
- Splits the transcript into chunks with `RecursiveCharacterTextSplitter`.
- Generates embeddings with OpenAI via `OpenAIEmbeddings` and stores them in `FAISS`.
- Retrieves relevant chunks for a question and prompts a `ChatOpenAI` model to answer using only the retrieved context.

**Repository files**

- File: [main.py](main.py) — core script that performs indexing and retrieval-augmented Q&A.

**Requirements**

- Python 3.10+
- Accounts / keys: OpenAI API key

Suggested Python packages (example):

```
youtube-transcript-api
langchain-core
langchain-text-splitters
langchain-openai
langchain-community
faiss-cpu
python-dotenv
openai
```

You can install these with a virtual environment. Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install youtube-transcript-api langchain-core langchain-text-splitters \
	langchain-openai langchain-community faiss-cpu python-dotenv openai
```

**Setup**

1. Create a `.env` file in the project root and add your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

2. (Optional) Edit `video_id` and `question` in `main.py` to point to the video you want to index and the question you want answered.

**Usage**
Run the main script:

```bash
python main.py
```

The script will:

- Fetch the transcript for the configured `video_id` (prints a message if captions are disabled).
- Create chunks, build embeddings, and store them in a FAISS vector store.
- Use the retriever + `ChatOpenAI` chain to answer the sample question and print the response.

**Notes & troubleshooting**

- If a video has captions disabled you will see `No captions available for this video.` and no indexing will occur.
- Embedding and chat calls use OpenAI — watch your API usage and rate limits.
- If you see import errors, ensure the package names and versions match your environment; some LangChain modules are modularized and package names can change.

**Next steps / ideas**

- Persist the FAISS index to disk and load it between runs.
- Add a small CLI to supply `video_id` and question at runtime.
- Add caching and batching for large transcripts.

**License**
This project is provided as-is for learning and demo purposes.
