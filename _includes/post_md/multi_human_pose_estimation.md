


## Application of 3D Human Pose Estimation
3D human pose estimation finds applications across various domains:

- **Motion Capture**: Used in movies, games, and VR for lifelike animations.
- **ControlNet for stable difusion**: Transfer character poses using ControlNet Stable Diffusion
- **Sports** Analysis: Enhances technique, prevents injuries in athletes.
- **Security**: Detects suspicious behavior in public spaces.
- **Human-Computer Interaction**: Enables gesture recognition in gaming and VR.

## Algorithm overview
Getting depth information from a single image is challenging due to the lack of depth cues.
One common solution for 3d human pose estimation is to use multi vew tringulation. The algorithm can be divided into the following steps: 
## 1 2d Human pose estimation 
1 top to bottom pipeline 
Using a top-to-bottom pipeline, the algorithm initially detects the human body to obtain a bounding box and then proceeds to detect the keypoints. 
Numerous real-time algorithms exist for 2D human pose estimation, such as Medias, HRNet, RTMpose, YOLO Pose, etc. Below is an example of 2D human pose estimation using HRNet:
![alt text](/img/posts/2d_img_keypoints.png "2d keypoints detection"  width="500")


## 2 Multi-view correspondences 
We need to find the correspondences between the keypoints of the human body in different views. We will take the geometry .We can formulate this problem by Affinity matrix, see more details in the paper.
based on the reprojection error.
![alt text](/img/posts/mv_correspondence.png "Cross-view matching"  width="500")

## 3 Robust tringualtion
After we find the correspondences between the keypoints of the human body in different views, we can use the triangulation method to get the 3D coordinates of the keypoints of the human body. Howerver, both 2d keypoint estimation and correspondences can be noisy, so we need to use the robust triangulation method(similar to RANSAC) to get the 3D coordinates of the keypoints of the human body.

## Demo
We can see the demo of the multi-view human pose estimation in the following video. We can also apply similar methodnology to the object 3d keypoint estimation, such as the football. You can see the 3d human skeleton and also the football animation in the video.


Watch the following video demonstration showcasing multi-view human pose estimation. Similar methodology can be applied to object 3D keypoint estimation, such as in the case of a football. The video demonstrates the 3D human skeleton and includes football animation.

<div class="col-sm mt-0 mt-md-0">
    {% include video.html path="/img/posts/mv_person_compressed.mp4" position="center"  %}
</div>