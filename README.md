# Point2CADGradioApp
Point2CAD Reconstruction Algorithm and Visualization

This GitHub repository hosts an application built for Gradio for visualizing 3d Model. The application reconstructs CAD models using Point2CAD with point clouds as an input. 

The Point2CAD algorithm is taken from https://www.obukhov.ai/point2cad BY by Yujia Liu, Anton Obukhov, Jan Dirk Wegner, and Konrad Schindler. 

The source code of the of the algorithm, assets, doc,  of the algorithm are from https://github.com/YujiaLiu76/point2cad .

POint2cad Implementation description : 
- Clusterization : Partition the point cloud into clusters corresponding to
the CAD model’s topological faces with pre trained model.
- Fit an analytical surface primitive to each cluster with geometric primitives and Freeform surfaces generated with INR.
- Find the effective area of each parametric surface and clip it, leaving enough margin to intersect adjacent surfaces.
- Perform pairwise surface intersections to obtain a set of topologically plausible object edges.
- Perform pairwise edge intersection to identify a set of topological corners. Clip edges based on proximity to the remaining surface regions and inferred corners.

## Command
docker build -t p2cad_app ./app/build

docker compose up

