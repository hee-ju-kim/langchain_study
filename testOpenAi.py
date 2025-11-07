from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Runnable

# 천문학 분야의 전문가로서 행동하라고 지시
# Question>: {input} 부분에서 실제 질문을 받아 답변하도록 요청
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")

# 모델의 출력을 문자열 형태로 파싱하여 최종 결과를 반환
output_parser = StrOutputParser()

# LCEL chaining
# | >> 앞 객체의 출력을 뒤 객체의 입력으로 연결해주는애
# prompt + model + output parser
# 프롬포트에 질문 입력해서 llm으로 넘긴다음 출력
chain = prompt | llm | output_parser

# chain 호출
res = chain.invoke({"input": "지구의 자전 주기는?"})

print(res)

# 1. 컴포넌트 정의
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. 체인 생성
chain = prompt | model | output_parser

# 3. invoke 메소드 사용
result = chain.invoke({"topic": "지구 자전"})
print("invoke 결과:", result)

# batch 메소드 사용
topics = ["지구 공전", "화산 활동", "대륙 이동"]
results = chain.batch([{"topic": t} for t in topics])
for topic, result in zip(topics, results):
    print(f"{topic} 설명: {result[:50]}...")  # 결과의 처음 50자만 출력

# stream 메소드 사용
stream = chain.stream({"topic": "지진"})
print("stream 결과:")
for chunk in stream:
    print(chunk, end="", flush=True)
print()