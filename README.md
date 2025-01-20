# django-llm-chat-proj

관련 튜토리얼

* [RAG #01. RAG 밑바닥부터 웹 채팅까지](https://ai.pyhub.kr/rag-01/)

## 가상환경 생성 및 활성화

```
python -m venv .venv         # 가상환경 생성

venv\Scripts\activate       # 윈도우
source ./.venv/bin/activate  # 맥/리눅스

python -m pip install -r requirements.txt
```

## 환경변수 설정

`.env` 파일을 생성하고 `OPENAI_API_KEY` 값을 설정해주세요.

```
cp .env.sample .env
```

## 데이터베이스 생성 및 슈퍼유저 생성

```
python manage.py migrate
python manage.py createsuperuser
```

## Vector Store 생성

```
python manage.py make_vector_store ./chat/assets/빽다방.txt
```

## 개발서버 구동

```
python manage.py runserver 0.0.0.0:8000
```

