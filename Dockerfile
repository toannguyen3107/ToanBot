FROM python:3.10-slim

WORKDIR /app

COPY requirement.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir data chroma_db

RUN python3 scrape_kali_tools.py

CMD ["python", "main.py"]