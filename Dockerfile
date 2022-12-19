FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ./app /app

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]