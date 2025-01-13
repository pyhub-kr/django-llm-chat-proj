import openai
from django.conf import settings
from . import rag


client = openai.Client(
    api_key=settings.OPENAI_API_KEY,
)


def make_ai_message(system_prompt: str, human_message: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_message},
        ],
    )
    ai_message = completion.choices[0].message.content

    return ai_message


class PaikdabangAI:
    def __init__(self):
        try:
            self.vector_store = rag.VectorStore.load(settings.VECTOR_STORE_PATH)
            print(f"Loaded vector store {len(self.vector_store)} items")
        except FileNotFoundError as e:
            print(f"Failed to load vector store: {e}")
            self.vector_store = rag.VectorStore()

    def __call__(self, question: str) -> str:
        search_doc_list = self.vector_store.search(question)
        지식 = "\n\n".join(doc.page_content for doc in search_doc_list)

        res = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"넌 AI Assistant. 모르는 건 모른다고 대답.\n\n[[빽다방 메뉴 정보]]\n{지식}",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            model="gpt-4o-mini",
            temperature=0,
        )
        ai_message = res.choices[0].message.content

        return ai_message


ask_paikdabang = PaikdabangAI()
