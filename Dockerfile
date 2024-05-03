FROM python:3.8-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app3.py"]

