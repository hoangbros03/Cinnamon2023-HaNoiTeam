FROM python:latest

ADD ../ .

RUN pip install -r requirements.txt

RUN pip install uvicorn

WORKDIR /api

RUN python3 file_download.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
