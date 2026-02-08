from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

#1. INDEXING
#DOC INGGESTION CODEEEE
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

#TEXT SPLIT
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
len(chunks)
chunks[100]

#EMBEDDING GENERATION AND STORING IN VEC STORE
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.index_to_docstore_id
vector_store.get_by_ids(['2436bdb8-3f5f-49c6-8915-0c654c888700'])

#2. Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever.invoke('What is deepmind'))

#3. Augmentation
model=ChatOpenAI()
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})



