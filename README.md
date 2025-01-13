# django-llm-chat-proj

## 가상환경 생성

```
python -m venv venv         # 가상환경 생성

venv\Scripts\activate       # 윈도우
source ./venv/bin/activate  # 맥/리눅스
```

## 환경변수 설정

`.env` 파일을 생성하고 `OPENAI_API_KEY` 값을 설정해주세요.

```
cp .env.sample .env
```

## 개발서버 구동

```
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 0.0.0.0:8000
```

