from langchain_community.document_loaders import TextLoader

loader = TextLoader('files/history.txt', encoding='utf-8')
data = loader.load()

# print(len(data[0].page_content))
# print(data[0].page_content)


# 각 문자를 구분하여 분할
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
  separator = '\n',
  chunk_size = 500,
  chunk_overlap  = 100,
  length_function = len,   
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts))
print(len(texts[0]), len(texts[1]), len(texts[2]))
print(texts[0])

# RecursiveCharacterTextSplitter > 내용기반으로 텍스트 분할
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts[0]), len(texts[1]), len(texts[2]))
print(texts[0])


# 토큰수로
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

docs = text_splitter.split_documents(data)
print(len(docs))

print(len(docs[1].page_content))
print(docs[1])