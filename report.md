# **CARVR: A Virtual Reality Implementation of Free Space Carving**

Ilya Nalivaiko  
University of Alberta  
nalivaik@ualberta.ca

# **Abstract**

	*Creating 3D models of the environment from live video has been a key goal of computer vision, but traditional approaches often fall short in speed or usability, or require expensive dedicated hardware. This report presents CARVR, an implementation of real-time spatial reconstruction via free space carving, utilizing the Meta Quest 3 VR headset as the capture and display device. The result is a metric scale 3D mesh created and displayed to the user in AR with minimal latency, and textured using the source camera frames. However, the accuracy of the reconstruction is hampered by the camera quality and resulting depth estimation. Despite this, the system presents a promising approach to using consumer VR / AR hardware for spatial mapping of indoor environments.*

# **1\. Introduction**

Creating detailed 3D models from a camera feed is a key goal in computer vision. Traditional SLAM / SFM algorithms work purely with images to estimate both the 3D world and the camera motion, which is computationally expensive and limits the accuracy. These systems tend to batch process pre-recorded video, as they are too slow for real time use. IMU sensors can provide real time instantaneous small motion estimation, but suffer from drift over long sequences. Calibrated 3D scanners are widely used in scientific and engineering applications, but they are generally cumbersome and prohibitively expensive. Most importantly, the above systems usually focus on creating a 3D point cloud, with a 3D mesh being created as an afterthought using a fitting algorithm like Poisson surfaces.   
	The Meta Quest 3 is a consumer VR headset with dual forward facing cameras, and sophisticated inside-out tracking that uses a combination of visual, IMU, and infrared sensors to provide a highly accurate position estimate. It also runs an Android-based operating system which, while somewhat locked down by Meta, provides a sufficient amount of freedom to make it a useful and accessible platform for computer vision. It is generally meant to be used indoors, where the overall shape of the captured model is expected to be concave. Free space carving \[1\] is an algorithm that creates a triangulated 3D mesh based on image feature observations and camera pose estimates, and is fast enough to run in real time. As such, it seems the perfect combination to create an accessible and easy to use system that creates detailed and metrically calibrated 3D scans of indoor environments in real time.  
	The paper is organized as follows: Section 2 provides an overview of the related work. The methods are described in sections 3, 4, and 5, where 3 is a high level overview of the system, 4 describes the key algorithmic process that turns the camera feed into a 3D model, and 5 describes how this is implemented in practice. Section 6 shows some experimental results that illustrate the key functionality. Section 7 then concludes the discussion of the project in its current state, and potential directions of future work are described in Section 8\.

# **2\. Related Work**

Most 3D reconstruction methods rely on feature matching or tracking. Shi-Tomasi corners \[4\] provide a reliable method for identifying potential features, while Kanade-Lucas-Tomasi’s (KLT) pyramidal tracker allows for efficient tracking over time \[3\]. These methods are faster than the descriptor matching typically used in batch processing, and are less prone to false matches. However, because continuous feature tracking can often fail, descriptor-based recovery methods are useful, using techniques like Lowe’s ratio test \[8\]. Since visual features often cluster in small, feature-rich regions, Teschner et al. \[5\] and Neubeck and Van Gool \[6\] developed non-maximal suppression methods using spatial hashing to ensure a uniform distribution.

A major architectural shift occurred with PTAM \[7\], which decoupled tracking and mapping. To estimate the depth of points, probabilistic depth filtering and Bayesian updates became the standard \[2, 9\]. While many real-time reconstruction systems stop at generating a point cloud, Lovi et al. \[1\] developed a method to extract triangulated meshes in real time using both the point data and camera visibility lines. Finally, projecting camera images to texture these meshes without creating visual artifacts is challenging. Waechter et al. \[10\] utilized graph cuts to assign the same camera to large, contiguous regions of the mesh, which improves visual consistency.

# **3\. System Overview**

The system is a real time spatial reconstruction pipeline designed for a stereo camera with known pose (in this case, a Quest 3 VR headset). An app running on the quest streams video from the two forward facing cameras over the network to a computer for processing. The PC side then generates a 3D triangulated mesh. This is broken across several nodes:

1. Receiver: Communicates with the Quest 3 to decode a camera feed with pose  
2. Spatial & tracking: Performs the bulk of the work in tracking and 3D position estimation, and publishes keyframes with triangulated keypoints visible in it.  
3. Carving: The interface for Delaunay freespace carving \[1\], creates a 3D mesh  
4. Streamer: Sends the resulting mesh back to the Quest for live feedback  
5. Saver: The mesh (3) and keyframes (2) are saved during runtime. They are used to texture the mesh and create a photorealistic reconstruction afterwards.

The mesh is displayed in the Quest overlaid on top of the real world passthrough. The textured mesh is saved as a standard obj file to be used for visualization after the capture is complete.

# **4\. Algorithms**

This section focuses on the theoretical implementation of the project. Specific implementation details are withheld and discussed in section 5\. In regards to the above overview, the receiver is described in section 4.1. The spatial node is of the most algorithmic interest, and is discussed in sections 4.2 to 4.4. The carving node is primarily a wrapper for David Lovi’s implementation \[1\], and an overview is given in section 4.5. The streamer and saver nodes are of no algorithmic interest, and as such are discussed in section 5 only. The texturing is discussed in section 4.6. Note that as it is extremely slow, the texturing step does not take place during runtime, only after the final mesh is saved, and as such it isn’t a node.

## **4.1 Image pre-processing**

Although the stereo feed is (mostly) synchronized between the cameras and the pose, it is **not** assumed to be rectified as-is. The first step of the process must therefore be to calibrate the cameras using a known calibration grid to acquire each camera’s calibration parameters, as well as a stereo rotation calibration. The resulting calibration matrices are then applied to the incoming images before being processed any further. The pose is similarly shifted to be the position of the left camera. From this point on the left camera is treated as the primary image feed, with the right being used for stereo depth estimation. Finally, although the Quest 3 has color passthrough, the image is actually converted to grayscale for the majority of the processing. The color image is saved only for visualization and the later texturing step.

## **4.2 Point Acquisition**

The stereo rectified grayscale images and pose are passed to the tracker. On initialization, it must first find points to track. The primary method of finding new points is Shi-Tomasi corners, a.k.a. Good Features to Track (GFTT) from \[4\]. However, the search space is first constrained by a mask that removes the edges of the image, and areas around existing points. The former is to ensure the entire patch around the potential point fits inside the frame, and the latter discourages clustering. Thus a set of new points is acquired.  
For each feature identified by GFTT in the left camera image, an epipolar stereo search is performed in the right image. Rather than relying blindly on the best match, a Zero-mean Normalized Cross Correlation (ZNCC) check ensures that the point has significant correlation to the feature in the left image, and that it is significantly higher than the second-best match (which may happen on repetitive patterns). Any points which could not be matched in the right image with a high degree of certainty are discarded. For each surviving point, we do not immediately trust the triangulation \[2\]. Rather, we save the point with a number of parameters:

1. The u,v coordinates it was found at  
2. The pose of the current camera (“birth pose”)  
3. The Kalman Depth Filter initialized to the stereo estimate depth and a high sigma  
4. The state ‘NEW’  
5. Its observation history, which is initialized to the current observation

	This information persists with the point across its lifetime. It is not tied to any given frame. However, more information may be attached to a point, as discussed in the next section.

## **4.3 Tracking**

	First, it is important to define the states a point may be in during its lifetime. At any stage in the process, the tracker has a set of points distinguished by the following:

1. NEW points are ones which have been recently initialized, and have a high variance depth estimate. They are candidate keypoints.  
2. MATURE points have been observed from a wide baseline, and have a strong depth estimate. A 3D position and a BRISK descriptor are computed for them.  
3. DEAD points are those that have failed to reach maturity before being lost or going out of frame. These points will never be tracked again, and are deleted.  
4. LOST points are previously MATURE points that have been lost or went out of frame. They are kept, and the tracker will attempt to re-establish their position.

The tracker works in several stages, which primarily update the position and status of the points.  
Points that are NEW or MATURE (observed in the previous frame) are tracked by pyramidal Kanade-Lucas-Tomasi (KLT) tracking \[3\]. If the point is tracked successfully, its image position (u,v) and patch is updated accordingly, otherwise it is set to DEAD or LOST as defined above.  
Points that are marked as LOST (whether in this or a previous frame) are attempted to be recovered. As KLT and ZNCC become unreliable in this situation, BRISK descriptors are used \[7\]. If a lost point should hypothetically be visible in frame (based on its 3D position and the camera frustum), a BRISK feature match will be performed using the Hamming distance of the descriptors. The search is restricted to a small region around the expected location for speed and accuracy. To further ensure the match is correct, both the absolute quality of the match, and Lowe’s ratio \[8\] to the second best match is used. If a good match is found, the point is promoted back to MATURE and its u,v position updated accordingly. Otherwise, it stays lost, and this repeats in subsequent frames.  
After tracking and re-acquisition is performed, the points may be clustered together in a small feature rich environment. A simple non maximal suppression (NMS) is performed using an O(1) spatial hashing approach \[5\]\[6\]. The coordinates are mapped into discrete buckets, and if several fall in the same bucket, the one with the lowest position variance survives. Points that get suppressed are demoted to LOST or DEAD according to their current maturity status. This ensures a uniform spatial distribution.  
	The filtering stage is where points are actually promoted from NEW to MATURE. When a significant change in the observed baseline (movement along camera plane) happens, we get another depth estimate using the stereo cameras, and a Kalman update is performed to shift the mean and shrink the variance \[2\]. The variance is proportional to the distance, and inversely proportional to the baseline and ZNCC match quality, meaning the ideal case is a wide baseline, high (near 1\) ZNCC match, and an overall close distance. Once the variance shrinks below a given threshold, the point is promoted to MATURE, its 3D pose triangulated, and a BRISK descriptor \[9\] is cached.  
	If, after the above process, the number of active points drops below a threshold, the acquisition process defined in 4.2 is repeated again, with existing points masking out regions of the search space to avoid clustering. Finally, DEAD points are permanently deleted to free memory. This concludes the cycle.

## **4.4 Keyframing**

	While the tracker runs on every frame (30fps), it would be too expensive to use all of them for carving. In reality, cameras that are close together and see mostly the same points are unlikely to provide new information. The spatial node declares a frame to be a keyframe any time rotational or translational movement exceeds a threshold. For a keyframe, the pose and the MATURE points visible in the frame are extracted and sent to CARV. While the incremental nature of CARV might suggest NEW points should also be used, and simply deleted if they are marked DEAD by the tracker, it proved too computationally heavy, and visually noisy considering the high churn of the tracker points. In practice, points mature fast enough for this to not be a significant issue for usability or latency, and provide a better geometry estimate. Separately, the keyframe pose and color image is also saved to disk to be used for texturing.

## **4.5 Free Space Carving**

	This section directly uses the CARV algorithm of \[1\]. The node receives the camera poses and keypoints visible in them, builds the visibility constraints, and triangulates the mesh. Each new point inside an existing tetrahedron becomes the corner of new 3D Delaunay tetrahedra filling the same space, and the visibility lines from cameras to points (corners) determine which should be carved away. To reduce noise, multiple visibility lines from some camera to some point must pass though for it to be “carved” (marked as free space).  
The resulting mesh is then extracted via a naive algorithm that defines the triangle face between a carved and uncarved tetrahedron to be a “surface” triangle. To reduce load, this is only done once per several keyframes, as they are sent too frequently in normal use to do for each. The output of the algorithm is a watertight triangulated mesh.  
![][image1]  
**Figure 4.1** *An illustration of free space carving from \[1\]. Each keyframe, step b. is performed to add new cameras and points. Steps c. and d. are delegated to every \~3-5 keyframes in practice to save on resources. The line between the free (white) and occupied (turquoise) tetrahedra, which in 3D would be a plane, is extracted as the final mesh.*

## **4.6 Display & Texturing**

	In real time use, the wireframe mesh from CARV is sent back to the Quest for immediate user feedback. As all points are natively in the OpenXR coordinate frame used by the quest positioning, they appear exactly on top of the real world where they were estimated to be. The points are drawn as a wireframe with semi-transparent triangle faces.  
	Texturing is done as a prost-processing step. For each triangle, the keyframe with the camera ray that most closely aligns the triangle normal is used. The color image is projected onto the triangle and that texture saved to a texture atlas. the result is a valid textured OBJ file, with metric scale coordinates and a reasonable rotation (level with the ground).

# **5 Implementation Details**

The Quest 3 runs a small app based on Meta’s OpenXR Passthrough example, which uses primarily native C++, with Java only serving as an app wrapper. The user sees a colored passthrough of the environment, and the front stereo camera feeds are captured using the API. A synchronized pose via OpenXR is tied to the camera images. The video is encoded with the hardware h.264 encoder, downscaled to 640x640 from the native 1280x1280. This is acceptable as the camera sensor isn’t high quality enough to justify the resolution, and using frequent I-frames and a high bitrate to eliminate encoder blocking artifacts is more important for tracking performance. The video is streamed over UDP to minimize latency and simplify the connection, as occasional frame drops are acceptable. The pose is attached to each frame using SEI NAL units, a part of the h.264 codec that allows for arbitrary per-frame information. As such, it is robust to out of order or dropped packets. Once the pipeline runs and generates the 3D mesh, it is sent back to the Quest over TCP to ensure safe arrival. It is then rendered on-device by the Quest 3 in the same app, overlayed on the passthrough video. As the mobile OpenGL ES does not have a native wireframe mode, the lines and occlusions are computed manually and applied with a simple shader.  
The PC processing side runs in a Docker / Podman container for ease of use, and relies on ROS 2 for inter-node communication. Each of the nodes described in section 2 are independent ROS nodes. As ROS topics natively handle buffering, the system is robust to any given part lagging behind momentarily \- the rest of the system is neither slowed down, nor is information dropped, it simply catches up when time allows. The freespace carving node, due to its heavy computations using CGAL and Eigen, is implemented in C++, and is primarily a wrapper around the existing open source code from \[1\]. The rest of the nodes are implemented in Python for faster iterative prototyping and to leverage available libraries. The receiver node uses multiprocessing rather than the threading library to bypass the GIL and decode the stereo feeds truly in parallel, and uses pyav (ffmpeg) software decoding as hardware proved too brittle for the networked use case. OpenCV implementations are used for most major algorithms (KLT, BRISK, calibration).

# **6 Results**

	The system works in practice, generating points in 3D space approximately near visual features and carving the space around them. The total time between looking at a new area and a mesh being generated is around a second provided the user walks at a normal pace. The system does require lateral movement while looking at features to properly triangulate them, however as vertical movement usually works just as well, squatting can effectively force a stationary triangulation.  
	The depth estimation of points often has significant error, floating above a surface or inside it by as much as 10 cm regularly. The mesh does not generally suffer from points above or in front of a surface, as free space carving can remove all tetrahedra spikes if other features are present. However, points inside a 3D object cause it to be carved away irreversibly. While the depth filtering and maturation steps help constrain this error, it is still notably present in the final version. Figure 6.1 shows an image from the VR view captured during an example run.  
The texturing step was done as a proof of concept rather than a final result, and as such should not be taken as a limitation of the meshing algorithm. While a proper approach would apply the camera image closest to the user’s current point of view, this was out of scope of the current project. A simple approach uses the best camera view for each triangle, which causes a discontinuous texture and looks visually unpleasant. An example of the same part of the lab shown in figure 6.1, as a statically textured model, is shown in figure 6.2. A more sophisticated approach, shown in figure 6.3, uses a graph cut to optimize the camera selection, favouring assigning the same camera to adjacent triangles \[10\] to produce a visually pleasing result. A view dependent texturing approach would dramatically improve the results, but would be incompatible with most traditional rendering software out of the box.

![][image2]  
	**Figure 6.1** *The mesh generated over a shelf surface. Points near the ground are too close to the user and create shapes that can be carved away later if more features on the bottom shelf are observed. Features near the pipe in the upper left shelf are too deep, and will cause irreversible structural errors. Also note that as the smooth white wall has no visual features, it is constructed by larger triangles anchored elsewhere.*

![][image3]  
	**Figure 6.2** *The scene reconstructed by texturing with a naive approach. The discontinuities in texturing are especially notable where there is significant error in triangulation and few keyframes to choose from. However, the table, shelf, and window are still distinctly recognizable despite the distortion.*

![][image4]  
**Figure 6.3** *The scene reconstructed by texturing with a graph cut approach. Since nearby regions are textured with the same camera view, the results are significantly better, though discontinuities are still visible. The Husky robot in the bottom right is notably distorted, but the table, window, and most of the shelf are visibly consistent. The chairs and shop vac on the floor are now visible, where in the previous approach they were not recognizable.*

# **7 Conclusion**

	The CARVR project successfully creates calibrated reconstructions of indoor environments using free space carving, and displays them to the user in AR in real time. This demonstrates a functional and accessible means to 3D scanning using consumer hardware. Approaches to texturing the resulting mesh using the same camera images used to generate it are discussed, and a simple approach is presented. Despite the challenges of working with the hardware outside its intended use case, this project shows using consumer VR headset for this purpose is feasible and promising.

# **8 Future Work**

This section describes many potential extensions and tangential directions that can build off the current state of the project. They are presented in no particular order.

Although the current implementation runs much of the tracking and geometry code on a PC, and uses the VR only as a camera and display device, the Quest 3 headset actually has significant computing power. It is therefore theoretically possible to run the entire system on device, though the ARM CPU and Android-based OS would make the port non-trivial.  
	KLT temporal tracking with an evolving patch is used for most of the process, with descriptor matching used only for recovery. This allows for sub-pixel accuracy and much higher speed, though it risks feature drift. In practice, this is generally a non-issue as only strong feature points are selected to begin with, but it may be useful to add an occasional descriptor or ZNCC match, with Lowe's ratio, against the birth patch to ensure no drift has occurred, and to correct it if it has. However, this is difficult in practice due to perspective warping.  
	The naive algorithm for mesh extraction is used in the CARV approach. The code also includes a max-flow min-cut algorithm, which claims to create smoother surfaces. This was experimented with but ultimately discarded as the result was too noisy to be usable, and the parameters were nearly impossible to tune in dynamic real time use. However, this or a related approach may be revisited later.  
	Using stereo and temporal matching for depth estimation proved difficult considering the quality of the forward facing cameras, as grain and noise would frequently distort features and there is no control over the auto exposure. The Quest 3 does have a built-in depth API, that uses both stereo matching and a currently unused infrared depth sensor to create a full depth image. Although restricted for “privacy” reasons, it may still prove useful if access can be gained. As the quality of the resulting mesh depends strongly on the precision with which 3D points can be estimated, any improvements here will lead to substantial improvements in the reconstruction.  
	The current system has no means to remove points from the carving engine, only to carve around them. This is because the MATURE points are ones the system is already certain of, and future tracking failures are attributed to occlusion or an issue with the tracker, not the point. In practice, this does not usually cause issues, as all tetrahedra around an occasional floating ghost point can be carved away. Issues can arise when the depth is overestimated, as the visibility constraint forces a hole to be carved unrecoverably. This is somewhat mitigated by requiring multiple visibility constraints before a tetrahedra is carved. However, some way to delete the wrong points entirely (potentially using the depth map above to check for occlusion, or assign a confidence score) may be of value, especially in changing environments.  
	The Quest uses inside-out tracking, meaning it relies on camera and IMU sensors only, without any fixed “lighthouse” markers. While this is a black box system, it is obvious that any such system must have some kind of loop closure algorithm to prevent drift. This would then require previous camera poses to be retroactively adjusted accordingly in the tracker. This implementation has no such adjustment and this does not appear to cause issues, but is something that should technically be addressed.  
Due to issues like motion blur and potential lag in the pose estimation, it is generally assumed that a camera image and pose will be more accurate and useful if taken while standing still. If accuracy is required, it may be worth switching from automatic keyframing to having the user stand still and press a button on the controller to capture a frame (tracking will still be done on the live video feed, else descriptor only methods will be required). However, such a system would cause user friction and be generally unintuitive for the average person.  
	The Quest also has controllers that are tracked accurately relative to the headset. While of little use to a vision only project, any space a controller is in must be unoccupied by other solid matter. Therefore, there is potential to somehow add the controller position as constraints alongside camera visibility to the carving algorithm. This also has potential use in robotics, where the robot’s position is known.  
	

# **Works Cited**

\[1\] Lovi, D., Birkbeck, N., Cobzaş, D., & Jägersand, M. (2010). Incremental Free-Space Carving for Real-Time 3D Reconstruction. *University of Alberta*.

\[2\] Forster, C., Pizzoli, M., & Scaramuzza, D. (2014). SVO: Fast Semi-Direct Monocular Visual Odometry. *IEEE International Conference on Robotics and Automation (ICRA)*.

\[3\] Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. *Proceedings of Imaging Understanding Workshop*.

\[4\] Shi, J., & Tomasi, C. (1994). Good Features to Track. *9th IEEE Conference on Computer Vision and Pattern Recognition*.

\[5\] Teschner, M., Heidelberger, B., Müller, M., Pomerantes, D., & Gross, M. (2003). *Optimized Spatial Hashing for Collision Detection of Deformable Objects.*

\[6\] Neubeck, A., & Van Gool, L. (2006). *Efficient Non-Maximum Suppression.*

\[7\] Klein, G., & Murray, D. (2007). *Parallel Tracking and Mapping for Small AR Workspaces (PTAM).*

\[8\] Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*.

\[9\] Civera, J., Davison, A. J., & Montiel, J. M. M. (2008). Inverse Depth Parametrization for Monocular SLAM. *IEEE Transactions on Robotics*.

\[10\] Waechter, M., Moehrle, N., & Goesele, M. (2014). Let There Be Color\! Large-Scale Texturing of 3D Reconstructions. European Conference on Computer Vision (ECCV).

[image1]: images/fig41.png

[image2]: images/fig61.png

[image3]: images/fig62.png

[image4]: images/fig63.png
