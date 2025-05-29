FROM python:3.10-slim

WORKDIR /app

COPY requirement.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]