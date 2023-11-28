# Segmentation-Project
Recent advancements in machine learning for medical purposes have placed significant emphasis on the automated segmentation of histopathological images. Our hypothesis is based on the idea that the biopsy tissue slides additional hematoxylin and eosin (H&E) stain, along with histological images variations in high-frequency patterns makes them well-suited for extracting features in the spectral domain [1][2]. These features can then be integrated with spatial domain features extracted by taking advantage of the attention map visualization obtained from a self-supervised pre-trained vision transformer (DINO V2) [3][4]. In this study, we introduce TF-Net, an architectural framework that combines frequency domain features extracted using spectral transformation with spatial domain information extracted using vision transformer to enhance the segmentation accuracy of histological images.

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
