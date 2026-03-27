xhost +local:
set -e
docker exec -it quest3_receiver /bin/bash build.sh
docker exec -it quest3_receiver /bin/bash run.sh