# RAG #02. 실전: 빽다방 문서 pgvector 임베딩

* 튜토리얼 : https://ai.pyhub.kr/rag-02/

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

## 개발서버 구동

```
python manage.py runserver 0.0.0.0:8000
```

