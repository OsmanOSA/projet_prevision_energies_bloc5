FROM python:3.10-slim 
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN apt-get update && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

CMD ["python3", "app.py"]

