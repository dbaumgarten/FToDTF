FROM python:3.5.5
ADD . /code
RUN mkdir /data && cd /code && pip3 install .
WORKDIR /data
ENTRYPOINT ["fasttext"]