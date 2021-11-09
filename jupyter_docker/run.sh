docker run \
-ti --rm --network=host \
--name jupyter-cont \
-v /etc/passwd:/etc/passwd \
-u $(id -u):$(id -g) \
-v $(realpath ..):/home/jovyan/work \
--gpus all \
jupyter-image \
jupyter-notebook --notebook-dir=/home/jovyan/work
#jupyter-notebook --config="/workspace/jupyter_notebook_config.py"
#start.sh jupyter notebook --NotebookApp.token=''
