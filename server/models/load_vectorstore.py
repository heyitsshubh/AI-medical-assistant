import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings



load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


print(
    "PINECONE KEY:",
    PINECONE_API_KEY
)
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medicalindex"

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# Lazy initialization of Pinecone
_pc_instance = None
_index_instance = None

def _initialize_pinecone():
    """Initialize Pinecone on first use (lazy loading)"""
    global _pc_instance, _index_instance
    
    if _pc_instance is not None:
        return _index_instance
    
    # create pinecone client
    _pc_instance = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    existing_indexes = [i["name"] for i in _pc_instance.list_indexes()]

    # determine embedding dimension by creating a quick test embedding
    # this ensures we create an index with the correct dimensionality
    embedding_model_name = "models/gemini-embedding-001"
    embed_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    try:
        sample_vec = embed_model.embed_query("test")
        desired_dim = len(sample_vec)
    except Exception:
        # fallback to a conservative default if embedding call fails
        desired_dim = 3072

    # if index exists, verify its dimension; if mismatch, delete & recreate
    if PINECONE_INDEX_NAME in existing_indexes:
        try:
            info = _pc_instance.describe_index(PINECONE_INDEX_NAME)
            current_dim = None
            # try several ways to read the dimension depending on SDK response
            if hasattr(info, "dimension"):
                current_dim = info.dimension
            elif hasattr(info, "spec") and hasattr(info.spec, "dimension"):
                current_dim = info.spec.dimension
            elif isinstance(info, dict) and "dimension" in info:
                current_dim = info["dimension"]
        except Exception:
            current_dim = None

        if current_dim is not None and int(current_dim) != int(desired_dim):
            print(f"Existing index dimension {current_dim} != desired {desired_dim}; recreating index")
            try:
                _pc_instance.delete_index(PINECONE_INDEX_NAME)
            except Exception:
                pass
            existing_indexes = [i["name"] for i in _pc_instance.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        _pc_instance.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=desired_dim,
            metric="dotproduct",
            spec=spec
        )
        while not _pc_instance.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    _index_instance = _pc_instance.Index(PINECONE_INDEX_NAME)
    return _index_instance


# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    index = _initialize_pinecone()
    embed_model = GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001",)
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")