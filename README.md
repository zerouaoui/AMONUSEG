# AMONuSeg: A Histological Dataset for African Multi-Organ Nuclei Semantic Segmentation.

Nuclei semantic segmentation is a key component for advancing machine learning and deep learning applications in digital pathology.
However, most existing segmentation models are trained and tested on
high-quality data acquired with expensive equipment, such as whole slide
scanners, which are not accessible to most pathologists in developing
countries. These pathologists rely on low-resource data acquired with
low-precision microscopes, smartphones, or digital cameras, which have
different characteristics and challenges than high-resource data. Therefore, there is a gap between the state-of-the-art segmentation models
and the real-world needs of low-resource settings. This work aims to
bridge this gap by presenting the first fully annotated African multiorgan dataset for histopathology nuclei semantic segmentation acquired
with a low-precision microscope. We also evaluate state-of-the-art segmentation models, including spectral feature extraction encoder and vision transformer-based models, and stain normalization techniques for
color normalization of Hematoxylin and Eosin-stained histopathology
slides. Our results provide important insights for future research on nuclei
histopathology segmentation with low-resource data.
# Project 
* Datasets Used : Crynosef - TNBC - Monuseg and Private African dataset 
* Models Baselines: UNET - YNET (CNN Blocks) - YNET (FFC Blocks) - DIANTE (Using DINO)
* Proposed Model: FTNET (FFC BLOCK + ViT blocks)

# Proposed Model ED
ongoing 
# Results
ongoing
# Requierement
ongoing
## References
[1] A. Farshad, Y. Yeganeh, P. Gehlbach, and N. Navab, “Y-Net: A Spatiospectral Dual-Encoder Network for Medical Image Segmentation.”

[2] L. Chi, B. Jiang, and Y. Mu, “Fast Fourier Convolution.”

[3] Y. Yeganeh, A. Farshad, P. Weinberger, S.-A. Ahmadi, E. Adeli, and N. Navab, “DIAMANT Dual Image-Attention Map Encoders For Medical Image Segmentation.”

[4] M. Caron et al., “Emerging Properties in Self-Supervised Vision Transformers.”
