from pathlib import Path

from django.core.management import BaseCommand
from tqdm import tqdm

from chat import rag
from chat.models import PaikdabangMenuDocument


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "txt_file_path",
            type=str,
            help="VectorStore로 저장할 원본 텍스트 파일 경로",
        )

    def handle(self, *args, **options):
        txt_file_path = Path(options["txt_file_path"])

        doc_list = rag.load(txt_file_path)
        print(f"loaded {len(doc_list)} documents")
        doc_list = rag.split(doc_list)
        print(f"split into {len(doc_list)} documents")

        # vector_store = rag.VectorStore.make(doc_list)
        # vector_store.save(settings.VECTOR_STORE_PATH)

        # 문서 목록을 순회하며, 모델 인스턴스를 생성하고 저장합니다.
        for doc in tqdm(doc_list):
            paikdabang_menu_document = PaikdabangMenuDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            paikdabang_menu_document.save()
