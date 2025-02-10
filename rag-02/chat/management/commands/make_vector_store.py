import sys
from pathlib import Path
from typing import Type

from django.core.management import BaseCommand
from django.db.models import Model
from django.utils.module_loading import import_string
from tqdm import tqdm

from chat import rag
from chat.models import Document


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "model",
            type=str,
            help="저장할 Document 모델 경로 (예: 'chat.PaikdabangMenuDocument')",
        )
        parser.add_argument(
            "txt_file_path",
            type=str,
            help="VectorStore로 저장할 원본 텍스트 파일 경로",
        )

    def print_error(self, msg: str) -> None:
        self.stdout.write(self.style.ERROR(msg))
        sys.exit(1)

    def get_model_class(self, model_path: str) -> Type[Model]:
        try:
            module_name, class_name = model_path.rsplit(".", 1)
            dotted_path = ".".join((module_name, "models", class_name))
            ModelClass: Type[Model] = import_string(dotted_path)
        except ImportError as e:
            self.print_error(f"{model_path} 경로의 모델을 임포트할 수 없습니다. ({e})")

        if not issubclass(ModelClass, Document):
            self.print_error("Document 모델을 상속받은 모델이 아닙니다.")
        elif ModelClass._meta.abstract:
            self.print_error("추상화 모델은 사용할 수 없습니다.")

        return ModelClass

    def handle(self, *args, **options):
        model_name = options["model"]
        txt_file_path = Path(options["txt_file_path"])

        ModelClass = self.get_model_class(model_name)

        doc_list = rag.load(txt_file_path)
        print(f"loaded {len(doc_list)} documents")
        doc_list = rag.split(doc_list)
        print(f"split into {len(doc_list)} documents")

        new_doc_list = [
            ModelClass(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in tqdm(doc_list)
        ]
        ModelClass.objects.bulk_create(new_doc_list)
