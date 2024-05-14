
## BEV percetion

## What is bevfusion 


Here's a simplified algorithm overview:

## BEVFusion Algorithm Overview
BEV Perception, or Bird's-Eye View Perception, is a critical technology in autonomous driving. It involves understanding the surrounding environment from a top-down perspective, similar to a map view. 
BEVFusion is a powerful framework for fusing **liDAR** and **camera** data for **3D object detection** in autonomous driving scenarios. It follows a Lift-Splat-Shoot paradigm to transform camera features into the BEV space. Here's a high-level overview of the process:
Lift: Camera features are "lifted" from the 2D image plane into 3D space using **depth estimation**. This is often done by predicting depth probability distributions for each pixel.
Splat: The lifted 3D features are "splatted" onto the BEV grid. Each feature contributes to multiple BEV cells based on its depth uncertainty. This accounts for potential errors in depth estimation.
Shoot: The splatted features are aggregated within each BEV cell, creating a dense representation of the scene from the camera's perspective.

<div>
<img src="../../../assets/posts/bev_overview.png" width="800" height: auto/>
</div>

## Deploy and demo image
The BEVFusion algorithm can be deployed on an NVIDIA GPU accerlated by CUDA and TensorRT and work as a ROS-node real-time inference. The following steps are involved in deploying the BEVFusion algorithm. Here's a demo showcasing the BEV perception system in action:

<div class="col-sm mt-0 mt-md-0">
    {% include video.html path="../../../assets/posts/bev_compressed.mp4" position="center"  %}
</div>


## Refrences:
> [BEVFusion Paper](https://arxiv.org/abs/2205.13542) \\
> [BEVFusion CUDA Deployment](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution)\\
> [nuScenes Dataset](https://www.nuscenes.org/)\\
> [Towards Viewpoint Robustness in Bird’s Eye View Segmentation]( https://nvlabs.github.io/viewpoint-robustness/)