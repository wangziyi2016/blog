


## Applications of 3D Human Pose Estimation
3D human pose estimation means determining the precise 3D coordinates of human joints from 2D images or videos. This technology is invaluable across various fields because it captures the complete spatial positioning of the human body, offering depth information that 2D methods simply cannot provide. This opens up a wealth of possibilities, including:
- **Motion Capture**: In film, gaming, and virtual reality, 3D pose estimation enables the creation of highly realistic animations by accurately tracking human movement in three dimensions.
- **ControlNet for Stable Diffusion**: ControlNet uses 3D pose estimation to guide the generation of images, allowing for precise manipulation and transfer of character poses.
- **Sports Analysis**: By providing detailed 3D representations of athletes' movements, coaches and trainers can analyze technique, identify areas for improvement, and help prevent injuries.
- **Human-Computer Interaction**: Gesture recognition systems in gaming and virtual reality benefit from 3D pose estimation, enabling more intuitive and natural interaction.


## Algorithm Overview
Obtaining depth information from a single image is challenging. One solution for 3D human pose estimation is multi-view triangulation. The algorithm consists of these steps:\\
**2D Human Pose Estimation**:  Using a top-down pipeline, the algorithm first detects the human body within an image to obtain a bounding box. It then proceeds to locate key points (joints) of the body. Numerous real-time algorithms exist for 2D pose estimation, including MediaPipe Pose, HRNet, RTMPose, and YOLO_Pose.
<img src="../../../assets/posts/2d_img_keypoints.png" width="800" height= auto alt="2D keypoints detection">
<span class="caption text-muted"> "2D Human Keypoint Detection" </span>

**Multi-view Correspondences**: Key points across different camera views must be matched before triangulation. This can be formulated as a matching problem using epipolar geometry and person re-identification techniques.
<img src="../../../assets/posts/mv_correspondence.png " width="800" height= auto alt="Cross-view matching">
<span class="caption text-muted"> "Multi-view Correspondences" </span>
**Robust Triangulation**: Once correspondences are established, triangulation is used to calculate the 3D coordinates of the key points. Due to potential noise in both 2D key point estimation and correspondence matching, robust triangulation methods (similar to RANSAC) are employed to ensure accuracy.

## Demo
The three images below are input streams from different cameras, and the upper portion of the video displays a 3D visualization of the human skeleton generated by our algorithm, along with the camera layout.

This same approach can also be applied to 3D keypoint estimation for objects, such as tracking a football's movement. The video provides a clear example, with the football's 3D position visible in the visualizer.

<div class="col-sm mt-0 mt-md-0">
    {% include video.html path="../../../assets/posts/mv_person_compressed.mp4" position="center"  %}
</div>

<span class="caption text-muted"> "3D Human and Object Pose estimation(click to play)" </span>


## References
> [Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views](https://arxiv.org/pdf/1901.04111) \\
> [EasyMocap](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html)\\
> [Autodistill Grounded SAM](https://github.com/autodistill/autodistill-grounded-sam)\\
> [Ultralytics YOLOv8 ](https://github.com/ultralytics/ultralytics)