# # 웹문서
# import bs4
# from langchain_community.document_loaders import WebBaseLoader

# # 여러 개의 url 지정 가능
# url1 = "https://blog.langchain.dev/customers-replit/"
# url2 = "https://blog.langchain.dev/langgraph-v0-2/"

# loader = WebBaseLoader(
#     web_paths=(url1, url2),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("article-header", "article-content")
#         )
#     ),
# )
# docs = loader.load()
# len(docs)

# print(docs[0])


# # 텍스트 문서
# from langchain_community.document_loaders import TextLoader

# loader = TextLoader('files/history.txt', encoding="utf-8")
# data = loader.load()

# print(type(data))
# print(len(data))
# print(data)

# # 디렉토리 폴더
# from langchain_community.document_loaders import DirectoryLoader

# loader = DirectoryLoader(path='./files/', glob='*.txt', loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'} )

# data = loader.load()

# len(data)
# print(data)


# # csv
# from langchain_community.document_loaders.csv_loader import CSVLoader

# loader = CSVLoader(file_path='files/한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949')
# data = loader.load()

# len(data)
# print(data)

# loader = CSVLoader(file_path='files/한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949',
#                    source_column='연도')

# data = loader.load()

# print(data[0])


# loader2 = CSVLoader(file_path='files/한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949',
#     csv_args={'delimiter': ','},  # 실제 CSV 구분자
# )

# data2 = loader2.load()
# print(data2)


# # pdf
# PyPDFLoader: 텍스트 기반 PDF에서 텍스트 추출
# from langchain_community.document_loaders import PyPDFLoader

# pdf_filepath = 'files/000660_SK_2023.pdf'
# loader = PyPDFLoader(pdf_filepath)
# pages = loader.load()

# print(len(pages))
# print(pages[10])


# https://github.com/oschwartz10612/poppler-windows/releases 에서 설치 후 bin폴더 환경변수에 추가해주기
# UnstructuredPDFLoader: 텍스트, 이미지, 표, 헤더 등 문서 구조 인식 + 필요시 OCR 처리
# from langchain_community.document_loaders import UnstructuredPDFLoader

# pdf_filepath = 'files/000660_SK_2023.pdf'

# # 전체 텍스트를 단일 문서 객체로 변환
# loader = UnstructuredPDFLoader(pdf_filepath, mode='elements')
# pages = loader.load()

# print(len(pages))
# print(pages[100])

# # PyMuPDFLoader: “빠르고 정확한 텍스트 추출기. PyPDFLoader보다 훨씬 좋음.”
# from langchain_community.document_loaders import PyMuPDFLoader

# pdf_filepath = 'files/000660_SK_2023.pdf'

# loader = PyMuPDFLoader(pdf_filepath)
# pages = loader.load()

# print(len(pages))
# print(pages[0].metadata)
# print(pages[0].page_content)

# # 온라인 pdf
# from langchain_community.document_loaders import OnlinePDFLoader

# # Transformers 논문을 로드
# loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
# pages = loader.load()

# print(len(pages))
# print(pages[0].page_content[:1000])

from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader('./files/')
data = loader.load()

print(len(data))