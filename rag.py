from dotenv import load_dotenv

load_dotenv()

# RAG
# **검색 증강 생성(Retrieval-Augmented Generation)**의 약자로, LLM이 답변을 생성할 때 자체 학습 데이터뿐만 아니라 외부 데이터베이스에서 관련 정보를 검색하여 함께 활용하는 기술

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from mongo import MongoDBVectorStore

# 데이터 로드(Load Data)
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url, header_template={"User-Agent": "MyApp/1.0"})

# 웹페이지 텍스트 -> Documents
docs = loader.load()

print('len(docs)', len(docs))
print('len(docs[0].page_content)', len(docs[0].page_content))
print('docs[0].page_content[5000:6000]', docs[0].page_content[5000:6000])


# 텍스트 분할(Text Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()


print('len(splits)', len(splits))
print('splits[10]', splits[10])
print('splits[10].page_content', splits[10].page_content)


# 인덱싱(Indexing) > 분할된 텍스트를 검색 가능한 형태로 만드는 단계
vectorstore = MongoDBVectorStore.from_texts(
    [doc.page_content for doc in splits],
    embedding=embedding
)

embeddings = embedding.embed_documents([doc.page_content for doc in splits])
texts = [doc.page_content for doc in splits]
vectorstore.add_texts(texts, embeddings)

docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
print(len(docs))
print(docs[0])


# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Rretriever
retriever = vectorstore.as_retriever()

# Combine Documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
rag_chain.invoke("격하 과정에 대해서 설명해주세요.")