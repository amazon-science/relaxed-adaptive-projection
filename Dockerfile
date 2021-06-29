FROM python:3

WORKDIR /usr/src

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python"]
