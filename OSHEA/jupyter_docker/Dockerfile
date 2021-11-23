FROM pytorch/pytorch:latest

RUN apt update
RUN apt install -y python3 
RUN apt install -y python-pip
RUN pip install notebook easydict matplotlib scipy numpy
RUN pip install sklearn
run mkdir -p /home/steven
run chmod 777 /home/steven
#ADD jupyter_notebook_config.py /workspace
