from dotenv import load_dotenv

load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,  HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
chat_prompt = ChatPromptTemplate.from_messages([
  ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
  ("user", "{user_input}"),
])

chain = chat_prompt | llm | StrOutputParser()
res = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print(res)

# 대화형 챗봇에 사용
chat_prompt2 = ChatPromptTemplate.from_messages(
  [
    SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."), # AI가 처음 등장할 때의 [시스템 역할] 지시문
    HumanMessagePromptTemplate.from_template("{user_input}"), # 사용자가 직접 입력하는 부분
  ]
)
chain2 = chat_prompt2 | llm | StrOutputParser()

res2 = chain2.invoke({"user_input": "태양계에서 가장 작은 행성은 무엇인가요?"})
print(res2)