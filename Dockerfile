#STEP 1: TAKING THE BASE IMAGE AND CHANGING WORK DIRECTORY
FROM python:3.8.0-slim
RUN apt-get update \&& apt-get install gcc -y \&& apt-get clean
WORKDIR /usr/src/app


#STEP 2: COPYING AND INSTALLING THE REQUIREMENTS
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

#STEP 3: COPYING EVEYTHING FROM MY PROJECT DIRECTORY 
COPY . .
EXPOSE 8080
ENTRYPOINT ["python3"]]
CMD ["gunicorn main:app --reload --host 0.0.0.0 --port 8080"]