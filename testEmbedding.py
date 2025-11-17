from dotenv import load_dotenv

load_dotenv()

# 임베딩(Embedding)은 텍스트 데이터를 숫자로 이루어진 벡터로 변환하는 과정

# 임베딩 메소드:
# embed_documents: 문서 객체의 집합을 입력으로 받아, 각 문서를 벡터 공간에 임베딩 > 주로 대량의 텍스트 데이터를 배치 단위로 처리할 때 사용
# embed_query: 단일 텍스트 쿼리를 입력으로 받아, 쿼리를 벡터 공간에 임베딩 > 주로 사용자의 검색 쿼리를 임베딩하여, 문서 집합 내에서 해당 쿼리와 유사한 내용을 찾아내는 데 사용

# 1. 문서와 쿼리를 임베딩 → 벡터 생성
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings_model = OpenAIEmbeddings()

documents = [
    '안녕하세요!',
    '어! 오랜만이에요',
    '이름이 어떻게 되세요?',
    '날씨가 추워요',
    'Hello LLM!'
]

embeddings = embeddings_model.embed_documents(documents)

# 문서와 임베딩을 같이 묶어두기
doc_with_embeddings = list(zip(documents, embeddings))

# print(len(embeddings), len(embeddings[0]))
# print(embeddings[0][:20])


embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')
# print(embedded_query[:5])
# print(embedded_query)


# 2. 코사인 유사도로 가장 관련 있는 문서 찾기
# embeddings와 embedded_query를 2차원 배열로 만들어 코사인 유사도 계산
similarities = cosine_similarity([embedded_query], embeddings)

# 가장 높은 유사도 인덱스
best_idx = np.argmax(similarities)
print("가장 유사한 문서:", documents[best_idx])
print("유사도 점수:", similarities[0][best_idx])

# 3. 그 문서를 기반으로 LLM에게 질문

model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

context = documents[best_idx]

print(context)
question = "첫인사를 하고 이름을 물어봤나요?"

template = "다음 문서를 보고 질문에 답해주세요:\n{context}\n질문: {question}"
prompt = ChatPromptTemplate.from_template(template)
input_dict = {
    "context": context,
    "question": question
}

# 메시지 리스트 생성
messages = prompt.format_prompt(**input_dict).to_messages()

# generate()는 [[BaseMessage]] 형태를 기대하므로 한 겹 더 감싸기
response = model.generate([messages])

# 결과 출력
print(response.generations[0][0].text)
