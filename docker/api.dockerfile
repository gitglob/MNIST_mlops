FROM python:3.9
WORKDIR /code
COPY ./requirements_api.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app

CMD ["uvicorn", "app.main_get_file:app", "--host", "0.0.0.0", "--port", "80"]
