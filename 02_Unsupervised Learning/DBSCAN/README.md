# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Overview & Introduction
DBSCAN is a density-based clustering algorithm that groups together points that are closely packed in areas of high density, while marking points in low-density regions as outliers. Unlike many clustering algorithms, it does not require specifying the number of clusters beforehand and can discover clusters of arbitrary shape.

**Role in Unsupervised Learning**:
DBSCAN addresses limitations of centroid-based methods like K-means by identifying clusters of arbitrary shapes and automatically detecting outliers. It's particularly valuable when dealing with data containing clusters of varying density, size, and shape.

### Historical Context
Developed in 1996 by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu at the University of Munich. It was introduced to overcome limitations of existing clustering methods, particularly for spatial database applications, and has since become one of the most cited clustering algorithms in scientific literature.

---

## Theoretical Foundations

### Conceptual Explanation
DBSCAN builds clusters based on two key concepts: (1) "density reachability," where points are connected if they are within a specified distance of each other, and (2) "density connectivity," where points may be connected through chains of density-reachable points. The algorithm identifies core points (in dense regions), border points (on the edge of dense regions), and noise points (isolated points).

### Mathematical Formulation
**Core Points**: A point p is a core point if at least minPts points (including p) are within distance ε of it:
$$ |N_\varepsilon(p)| \geq \text{minPts} $$
where $N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \varepsilon\}$ is the ε-neighborhood of point p.

**Directly Density-Reachable**: A point q is directly density-reachable from a point p if:
1. $p$ is a core point
2. $q \in N_\varepsilon(p)$

**Density-Reachable**: A point q is density-reachable from a point p if there is a sequence of points $p_1, p_2, ..., p_n$ with $p_1 = p$ and $p_n = q$ where each $p_{i+1}$ is directly density-reachable from $p_i$.

**Density-Connected**: Two points p and q are density-connected if there exists a point o such that both p and q are density-reachable from o.

**Cluster Definition**: A cluster is a maximal set of density-connected points.

### Assumptions
1. Clusters are dense regions separated by low-density regions
2. Distances between points are meaningful (appropriate distance metric)
3. The density of clusters is consistent within each cluster
4. The appropriate values
