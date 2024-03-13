from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

#PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_API_KEY = "4727f843-5433-4194-8bb9-cc8545df365b"
PINECONE_API_ENV = "gcp-starter"

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


index_name="chatbot3"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)