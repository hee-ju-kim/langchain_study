from dotenv import load_dotenv
import os

load_dotenv()

print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")
print(f"[LANGCHAIN_TRACING_V2]\n{os.environ['LANGCHAIN_TRACING_V2']}")
print(f"[LANGCHAIN_ENDPOINT]\n{os.environ['LANGCHAIN_ENDPOINT']}")
print(f"[LANGCHAIN_PROJECT]\n{os.environ['LANGCHAIN_PROJEC']}")
print(f"[LANGCHAIN_API_KEY]\n{os.environ['LANGCHAIN_API_KEY']}")

import langchain
from langchain_openai import ChatOpenAI

print(f"LangChain 버전: {langchain.__version__}")

# 간단한 테스트
try:
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("안녕하세요!")
    print("✅ 설치 완료 - 정상 작동")
    print(f"응답: {response.content}")
except Exception as e:
    print(f"❌ 설정 오류: {e}")