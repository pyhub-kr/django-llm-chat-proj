import logging
import time
from typing import List, Iterable, Union

import openai
import tiktoken
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core import checks
from django.core.exceptions import ImproperlyConfigured, FieldDoesNotExist
from django.db import models
from django.db.models import Index
from django_lifecycle import hook, BEFORE_CREATE, BEFORE_UPDATE, LifecycleModelMixin
from pgvector.django import (
    VectorField,
    HnswIndex,
    CosineDistance,
    L2Distance,
    HalfVectorField,
    IvfflatIndex,
)

from chat.utils import make_groups_by_length
from chat.validators import MaxTokenValidator


logger = logging.getLogger(__name__)


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


class DocumentQuerySet(models.QuerySet):

    def bulk_create(self, objs, *args, max_retry=3, interval=60, **kwargs):
        # 임베딩 필드가 지정되지 않은 인스턴스만 추출
        non_embedding_objs = [obj for obj in objs if obj.embedding is None]

        # 임베딩되지 않은 인스턴스가 있으면, 해당 인스턴스들에 대해서만 임베딩 벡터 생성
        if len(non_embedding_objs) > 0:

            # 임베딩된 벡터를 저장할 리스트
            embeddings = []

            groups = make_groups_by_length(
                # 임베딩을 할 문자열 리스트
                text_list=[obj.page_content for obj in non_embedding_objs],
                # 그룹의 최대 허용 크기 지정
                group_max_length=self.model.embedding_max_tokens_limit,
                # 토큰 수 계산 함수
                length_func=self.model.get_token_size,
            )

            # 토큰 수 제한에 맞춰 묶어서 임베딩 요청
            for group in groups:
                for retry in range(1, max_retry + 1):
                    try:
                        embeddings.extend(self.model.embed(group))
                        break
                    except openai.RateLimitError as e:
                        if retry == max_retry:
                            raise e
                        else:
                            msg = "Rate limit exceeded. Retry after %s seconds... : %s"
                            logger.warning(msg, interval, e)
                            time.sleep(interval)

            for obj, embedding in zip(non_embedding_objs, embeddings):
                obj.embedding = embedding

        return super().bulk_create(objs, *args, **kwargs)

    # TODO: 비동기 버전 지원
    async def abulk_create(self, objs, *args, max_retry=3, interval=60, **kwargs):
        raise NotImplementedError
        return await super().abulk_create(objs, *args, **kwargs)

    async def search(self, question: str, k: int = 4) -> List["Document"]:
        question_embedding: List[float] = await self.model.aembed(question)

        qs = None
        index: Index
        for index in self.model._meta.indexes:
            if "embedding" in index.fields:
                # vector_cosine_ops, halfvec_cosine_ops, etc.
                if any("_cosine_ops" in cls for cls in index.opclasses):
                    qs = (qs or self).annotate(
                        distance=CosineDistance("embedding", question_embedding)
                    )
                    qs = qs.order_by("distance")
                # vector_l2_ops, halfvec_l2_ops, etc.
                elif any("_l2_ops" in cls for cls in index.opclasses):
                    qs = (qs or self).annotate(
                        distance=L2Distance("embedding", question_embedding)
                    )
                    qs = qs.order_by("distance")
                else:
                    raise NotImplementedError(
                        f"{index.opclasses}에 대한 검색 구현이 필요합니다."
                    )

        if qs is None:
            raise ImproperlyConfigured(
                f"{self.model.__name__} 모델에 embedding 필드에 대한 인덱스를 추가해주세요."
            )

        return await sync_to_async(list)(qs[:k])

    def __repr__(self):
        return repr(list(self))  # QuerySet을 리스트처럼 출력


class Document(LifecycleModelMixin, models.Model):
    openai_api_key = settings.RAG_OPENAI_API_KEY
    openai_base_url = settings.RAG_OPENAI_BASE_URL
    embedding_model = settings.RAG_EMBEDDING_MODEL
    embedding_dimensions = settings.RAG_EMBEDDING_DIMENSIONS
    embedding_max_tokens_limit = settings.RAG_EMBEDDING_MAX_TOKENS_LIMIT

    page_content = models.TextField(
        validators=[MaxTokenValidator(embedding_model)],
    )
    metadata = models.JSONField(default=dict)
    embedding = VectorField(dimensions=embedding_dimensions, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # .as_manager() 메서드를 통해 모델 매니저를 생성하여
    # 디폴트 모델 매니저를 커스텀 쿼리셋으로 교체합니다.
    objects = DocumentQuerySet.as_manager()

    def __repr__(self):
        return f"Document(metadata={self.metadata}, page_content={self.page_content!r})"

    def __str__(self):
        return self.__repr__()

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
    def embed(
        cls, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        client = openai.Client(api_key=cls.openai_api_key, base_url=cls.openai_base_url)
        response = client.embeddings.create(
            input=input,
            model=cls.embedding_model,
        )
        if isinstance(input, str):
            return response.data[0].embedding
        return [v.embedding for v in response.data]

    @classmethod
    async def aembed(
        cls, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        client = openai.AsyncClient(
            api_key=cls.openai_api_key, base_url=cls.openai_base_url
        )
        response = await client.embeddings.create(
            input=input,
            model=cls.embedding_model,
        )
        if isinstance(input, str):
            return response.data[0].embedding
        return [v.embedding for v in response.data]

    @classmethod
    def get_token_size(cls, text: str) -> int:
        encoding: tiktoken.Encoding = tiktoken.encoding_for_model(cls.embedding_model)
        token: List[int] = encoding.encode(text or "")
        return len(token)

    @classmethod
    def check(cls, **kwargs):
        embedding_field_name = "embedding"

        errors = super().check(**kwargs)

        def add_error(msg: str, hint: str = None):
            errors.append(checks.Error(msg, hint=hint, obj=cls))

        embedding_dimensions: int = getattr(cls, "embedding_dimensions", None)
        if embedding_dimensions is None:
            add_error(
                "embedding_dimensions 클래스 변수가 누락되었습니다.",
                hint="embedding_dimensions 설정을 추가해주세요.",
            )

        try:
            embedding_field = cls._meta.get_field(embedding_field_name)
        except FieldDoesNotExist:
            add_error(
                f"{embedding_field_name} 필드가 누락되었습니다.",
                hint="embedding 필드를 추가해주세요.",
            )
        else:
            if embedding_dimensions > 2000:
                if not isinstance(embedding_field, HalfVectorField):
                    add_error(
                        f"{embedding_dimensions} 차원은 {embedding_field.__class__.__name__}에서 지원하지 않습니다.",
                        hint=f"{embedding_dimensions} 차원 이상은 HalfVectorField를 사용해주세요.",
                    )

            for index in cls._meta.indexes:
                if embedding_field_name in index.fields:
                    if isinstance(index, (HnswIndex, IvfflatIndex)):
                        if embedding_dimensions <= 2000:
                            for opclass_name in index.opclasses:
                                if "halfvec_" in opclass_name:
                                    add_error(
                                        f"{embedding_field.name} 필드는 {embedding_field.__class__.__name__} 타입으로서 "
                                        f"{opclass_name}를 지원하지 않습니다.",
                                        hint=f"{opclass_name.replace('halfvec_', 'vector_')}로 변경해주세요.",
                                    )
                        else:
                            for opclass_name in index.opclasses:
                                if "vector_" in opclass_name:
                                    add_error(
                                        f"{embedding_field.name} 필드는 {embedding_field.__class__.__name__} 타입으로서 "
                                        f"{opclass_name}를 지원하지 않습니다.",
                                        hint=f"{opclass_name.replace('vector_', 'halfvec_')}로 변경해주세요.",
                                    )
                    else:
                        add_error(
                            f"Document 모델 check 메서드에서 {index.__class__.__name__}에 대한 확인이 누락되었습니다.",
                            hint=f"{index.__class__.__name__} 인덱스에 대한 check 루틴을 보완해주세요.",
                        )

        return errors

    class Meta:
        abstract = True


class PaikdabangMenuDocument(Document):
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


class StarbucksMenuDocument(Document):
    class Meta:
        indexes = [
            HnswIndex(
                name="starbucks_menu_doc_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


class TaxLawDocument(Document):
    # category = models.ForeignKey(...)

    embedding_model = "text-embedding-3-large"
    embedding_dimensions = 1536 * 2

    embedding = HalfVectorField(dimensions=embedding_dimensions, editable=False)

    class Meta:
        indexes = [
            HnswIndex(
                name="taxlaw_doc_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                # halfvec 타입용 코사인 거리 연산 클래스
                opclasses=["halfvec_cosine_ops"],
            ),
        ]
