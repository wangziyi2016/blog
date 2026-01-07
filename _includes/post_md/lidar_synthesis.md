


#### Long-Tail Rare Objects Are a Major Challenge in AV Perception
One of the core challenges in autonomous driving is handling the long-tail distribution of objects and scenarios. While common classes such as cars, pedestrians, and cyclists appear frequently in most driving datasets, rare and unusual cases—like pedestrians pushing shopping trolleys, stray animals, traffic cones, or people carrying large objects—occur very infrequently in real-world data.
<img src="../../../assets/posts/nuscenes_long_tail.jpg" width="800" height= auto>
<span class="caption text-muted"> "Example of rare classes in NuScenes dataset: Traffic cone and pedestrian pushing shopping trolleys" </span>


#### LiDARsim: Bridging the Gap with Synthetic Data
To address this data scarcity, Uber ATG introduced [LiDARsim (2020)](https://arxiv.org/abs/2006.09348), a framework designed to tackle the long-tail problem through realistic data synthesis.
The algorithm follows a three-stage pipeline to generate high-fidelity training data:\\
1) **Mesh Generation**: Convert real-world environments and objects from sparse point clouds into dense 3D meshes.\\
2) **Scene Composition**: Procedurally compose new scenes by placing dynamic objects into static backgrounds.\\
3) **LiDAR Point Cloud Simulation**: Generate realistic LiDAR point clouds using raycasting physics.
<img src="../../../assets/posts/lidarsim_steps.png" width="800" height=auto>
<span class="caption text-muted"> "Steps for generating synthetic data with LiDARsim" </span>

#### Build a Catalog of 3D Mesh Assets from Real Point Cloud Data
<img src="../../../assets/posts/steps_generating_mesh.png" width="800" height= auto>
<span class="caption text-muted"> "Steps to generate mesh from real data" </span>
To create realistic scenes, LiDARsim reconstructs objects directly from LiDAR scans.
The figure above illustrates the overall pipeline for generating 3D meshes from real-world data.
- For each frame, objects are cropped from the point cloud using an object detection or LiDAR segmentation model.
- Building a complete 3D mesh from sparse LiDAR scans is challenging due to object motion and partial observations caused by occlusion. To address this, LiDARsim accumulates LiDAR points within the object’s bounding box and transforms them into a consistent object-centric coordinate system based on the bounding box center.
- Finally, outliers are removed and the object mesh is reconstructed using surfel-disk–based surface reconstruction.

Beyond the method proposed in the original paper, we can also use GenAI tools—such as [MeshyAI](https://www.meshy.ai/discover) and [InstantMesh](https://github.com/TencentARC/InstantMesh)—to generate meshes from real data. These tools are capable of converting 2D images into 3D meshes. However, post-processing is often needed to correct the **scale** of the generated mesh. Additionally, AI-generated meshes frequently contain too many triangles (high polygon count), requiring **mesh decimation** to reduce the number of triangles. This step is critical because meshes with excessive triangles increase computational costs and are unsuitable for efficient LiDAR simulation.
<img src="../../../assets/posts/meshyai_trolley_cone.png" width="800" height= auto>
<span class="caption text-muted"> "Example of mesh generated with GenAI tools: pedestrian pushing shopping trolleys and traffic cone" </span>

##### Composing Scenes for Simulation and LiDAR Point Cloud Generation
Once the asset catalog has been constructed, the system can synthesize new simulation scenarios using the following steps:
- Picking a static background and placing dynamic objects (e.g., vehicles, pedestrians, or rare classes) at plausible positions.
- Configuring the LiDAR sensor trajectory (e.g., where the ego vehicle is driving)
- Raycasting the mesh to generate the LiDAR point cloud.

<img src="../../../assets/posts/compose_scenes.png" width="800" height= auto>
<span class="caption text-muted"> "Example of synthetic point cloud generation with LiDARsim" </span>