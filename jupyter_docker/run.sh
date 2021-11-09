docker run \
-ti --rm --network=host \
--name jupyter-cont \
-v /etc/passwd:/etc/passwd \
-u $(id -u):$(id -g) \
-v $(realpath ..):/home/jovyan/work \
jupyter-image \
start.sh jupyter notebook --NotebookApp.token=''
