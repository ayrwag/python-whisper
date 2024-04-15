FROM huggingface/transformers-pytorch-gpu

WORKDIR /app

ENV PORT=8080

COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Expose ports
EXPOSE 8080

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]