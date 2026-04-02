# ROI-based image regionally blurred dataset

This repository is an anonymous supplementary material prepared for the double-blind review process of ACM Multimedia (ACM MM).

### Directory Structure

For each dataset sample, the data is organized as follows:

- `processed/`: Contains the images processed by our hybrid resolution compression algorithm. The selected ROI maintains the original clarity, while the background is downsampled (e.g., to 1/4 resolution) and then resized back to preserve the original image dimensions.
- `labeled/`: Contains visualization images. The **red boxes** indicate the ground-truth object bounding boxes, and the **thick green box** highlights the selected Region of Interest (ROI) that is preserved at high resolution.

## Data Availability Statement

> **Note:** Due to the strict double-blind review policy of ACM MM, only a representative subset of the processed datasets is provided in this repository for visualization and verification purposes.
> 
> **The full datasets, comprehensive benchmarks, and complete source code will be made entirely open-source and publicly available immediately upon the acceptance of this paper.**
