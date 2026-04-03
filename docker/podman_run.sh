# First, ensure the host allows the connection
xhost +local:$(whoami)

# Run the container with authority mounting
podman run -it --rm \
  --name quest3_receiver \
  --network host \
  --ipc=host \
  --userns=keep-id \
  --security-opt label=disable \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v $XAUTHORITY:/tmp/.Xauthority:Z \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/workspace:/workspace:Z \
  --device /dev/dri:/dev/dri \
  --replace \
  quest3_receiver:latest \
  /bin/bash