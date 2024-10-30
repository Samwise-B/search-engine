
FROM pytorch/pytorch


WORKDIR /workspace


COPY ./api/requirements.txt /requirements.txt


RUN pip install --no-cache-dir --upgrade -r /requirements.txt


COPY ./api ./api
COPY ./networks ./networks
#COPY ./weights ./weights
COPY ./utils ./utils
COPY ./dictionaries ./dictionaries
#COPY ./data ./data


CMD ["fastapi", "run", "api/main.py", "--port", "80"]