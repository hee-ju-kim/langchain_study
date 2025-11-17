from dotenv import load_dotenv
import os

load_dotenv()

import pymongo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

# MongoDB 벡터 저장소 클래스 정의
class MongoDBVectorStore(VectorStore):
    def __init__(self, embedding: OpenAIEmbeddings = None):
        """MongoDB 연결 및 컬렉션 초기화"""
        self.client = pymongo.MongoClient(os.environ['DATABASE'])
        self.db = self.client['langchain']
        self.collection = self.db['vectors']
        self.embedding = embedding

        
    @classmethod
    def from_texts(cls, texts: list, embedding: OpenAIEmbeddings, metadatas: list = None):
        """텍스트와 임베딩을 MongoDB에 저장하고 MongoDBVectorStore 인스턴스를 반환"""
        vectorstore = cls(embedding)
        embeddings = embedding.embed_documents(texts)
        vectorstore.add_texts(texts, embeddings, metadatas)
        return vectorstore

    def add_texts(self, texts: list, embeddings: list, metadatas: list = None):
        """텍스트와 임베딩을 MongoDB에 저장"""
        for text, embedding, metadata in zip(texts, embeddings, metadatas or [{}] * len(texts)):
            document = {
                "embedding": embedding,
                "metadata": metadata,
                "text": text
            }
            self.collection.insert_one(document)

    def similarity_search(self, query, k=5):
      """
      query: 텍스트 문자열 또는 이미 벡터화된 리스트
      """
      # 문자열이면 벡터로 변환
      if isinstance(query, str):
          # 클래스 내부에 이미 OpenAIEmbeddings 객체가 있으면 사용
          if not hasattr(self, 'embedding') or self.embedding is None:
              raise ValueError("MongoDBVectorStore 객체에 'embedding' 속성이 필요합니다. 생성 시 embedding 객체를 전달하세요.")
          query_vector = self.embedding.embed_query(query)
      else:
          query_vector = query  # 이미 벡터이면 그대로 사용

      # MongoDB에서 모든 문서 가져오기
      stored_embeddings = list(self.collection.find())
      similarities = []

      for doc in stored_embeddings:
          stored_embedding = np.array(doc["embedding"]).reshape(1, -1)
          query_embedding_array = np.array(query_vector).reshape(1, -1)
          similarity = cosine_similarity(query_embedding_array, stored_embedding)
          similarities.append((doc, similarity[0][0]))

      # 유사도 기준으로 정렬 후 k개 반환
      similarities.sort(key=lambda x: x[1], reverse=True)
      return similarities[:k]

    def as_retriever(self):
        """retriever 형태로 반환"""
        return self

    def retrieve(self, query_embedding, k=5):
        return self.similarity_search(query_embedding, k)
