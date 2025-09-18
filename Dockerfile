FROM python:3.10-slim

COPY ./app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE $PORT

CMD ["python", "app.py"]