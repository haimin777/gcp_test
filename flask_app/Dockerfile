FROM python:3.7.5
RUN apt-get update -y && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx

#RUN groupadd flaskgroup && useradd -m -g flaskgroup -s /bin/bash flask
#RUN echo "flask ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN mkdir -p /home/flask/app/web && mkdir -p /home/flask/app/web/results && mkdir -p /home/flask/app/web/templates
WORKDIR /home/flask/app/web
COPY requirements.txt /home/flask/app/web
COPY tools.py /home/flask/app/web
COPY utils.py /home/flask/app/web


COPY templates/* /home/flask/app/web/templates/
RUN pip3 install -r requirements.txt
COPY app.py /home/flask/app/web
COPY utils.py /home/flask/app/web
EXPOSE  8501
CMD ["flask", "run", "--host", "0.0.0.0"]