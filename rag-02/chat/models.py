from django.conf import settings
from django.db import models
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


class PaikdabangMenuDocument(models.Model):
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
