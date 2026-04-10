# CARVR

## What is this

This is my final project for CMPUT 428 - Computer Vision, at the University of Alberta. It uses the Quest 3 VR headset to create a virtual representation of a 3D environment using some of the techniques learned in class. The full report is under report.md

## Usage

### Quest app:

1. Connect the Quest 3 to the PC and enable ADB

2. Run the app located at `Meta-OpenXR-SDK/Samples/XrSamples/XrPassthrough/Projects/Android/build.gradle` in Android Studio. You may need to enable the NDK manually.

3. Run the app. It should ask for a camera permission but if it doesn't, manually run

```
adb shell pm grant com.oculus.xrpassthrough horizonos.permission.HEADSET_CAMERA
```

### On the PC:

1. Build the Docker/Podman container and run it with the scripts under `docker/`

2. Run `./build.sh` to build the ROS nodes. The first compile may take up to a minute, this is normal

3. Run the code with `./run.sh`

### Back to the Quest

Wear it and look around! You should see the mesh be generated in real time. For best results:

1. Try to keep your vision locked to an area as you move laterally (side to side or vertically)

2. Ensure you are in a well lit, feature rich environment

3. Avoid drastic shadows, reflections, bloom, or other lighting artifacts

### Finally, texturing

Once you are done, terminate the app with a single Ctrl-C. It will take a bit to fully shut down as it saves the final mesh.

Run the texturer under `/texture_maker/` by creating the virtual environment with required dependencies, then running the python script.
