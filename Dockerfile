FROM python:3.12.9

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app

CMD streamlit run app.py --server.port 8080 --server.enableCORS false