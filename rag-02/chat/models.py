from typing import List

import openai
from django.conf import settings
from django.db import models
from django_lifecycle import hook, BEFORE_CREATE, BEFORE_UPDATE, LifecycleModelMixin
from pgvector.django import VectorField, HnswIndex

from chat.validators import MaxTokenValidator


class Item(models.Model):
    embedding = VectorField(dimensions=3, editable=False)

    class Meta:
        indexes = [
            # https://github.com/pgvector/pgvector?tab=readme-ov-file#index-options
            HnswIndex(
                name="item_embedding_hnsw_idx",  # 유일한 이름이어야 합니다.
                fields=["embedding"],
                # 각 벡터를 연결할 최대 연결수
                # 높을수록 인덱스 크기가 커지며 더 긴 구축시간, 더 정확한 결과
                m=16,  # default: 16
                # 인덱스 구축시 고려할 후보 개수
                ef_construction=64,  # default: 64
                # 인덱스 생성에 사용할 벡터 연산 클래스
                opclasses=["vector_cosine_ops"],
            ),
        ]


class PaikdabangMenuDocument(LifecycleModelMixin, models.Model):
    openai_api_key = settings.RAG_OPENAI_API_KEY
    openai_base_url = settings.RAG_OPENAI_BASE_URL
    embedding_model = settings.RAG_EMBEDDING_MODEL
    embedding_dimensions = settings.RAG_EMBEDDING_DIMENSIONS

    page_content = models.TextField(
        validators=[MaxTokenValidator(embedding_model)],
    )
    metadata = models.JSONField(default=dict)
    embedding = VectorField(dimensions=embedding_dimensions, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def update_embedding(self, is_force: bool = False) -> None:
        # 강제 업데이트 혹은 임베딩 데이터가 없는 경우에만 임베딩 데이터를 생성합니다.
        if is_force or self.embedding is None:
            self.embedding = self.embed(self.page_content)

    @hook(BEFORE_CREATE)
    def on_before_create(self):
        # 생성 시에 임베딩 데이터가 저장되어있지 않으면 임베딩 데이터를 생성합니다.
        self.update_embedding()

    @hook(BEFORE_UPDATE, when="page_content", has_changed=True)
    def on_before_update(self):
        # page_content 변경 시 임베딩 데이터를 생성합니다.
        self.update_embedding(is_force=True)

    @classmethod
    def embed(cls, input: str) -> List[float]:
        """
        주어진 문자열에 대한 임베딩 벡터를 생성합니다.
        """
        client = openai.Client(api_key=cls.openai_api_key, base_url=cls.openai_base_url)
        response = client.embeddings.create(
            input=input,
            model=cls.embedding_model,
        )
        return response.data[0].embedding

    @classmethod
    async def aembed(cls, input: str) -> List[float]:
        client = openai.AsyncClient(
            api_key=cls.openai_api_key, base_url=cls.openai_base_url
        )
        response = await client.embeddings.create(
            input=input,
            model=cls.embedding_model,
        )
        return response.data[0].embedding

    class Meta:
        indexes = [
            HnswIndex(
                name="paikdabang_menu_doc_idx",  # 데이터베이스 내에서 유일한 이름이어야 합니다.
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]
