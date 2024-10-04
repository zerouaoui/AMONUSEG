# AMONuSeg: A Histological Dataset for African Multi-Organ Nuclei Semantic Segmentation.

Nuclei semantic segmentation is a key component for advancing machine learning and deep learning applications in digital pathology. However, most existing segmentation models are trained and tested on
high-quality data acquired with expensive equipment, such as whole slide scanners, which are not accessible to most pathologists in developing countries. These pathologists rely on low-resource data acquired with
low-precision microscopes, smartphones, or digital cameras, which have different characteristics and challenges than high-resource data. Therefore, there is a gap between the state-of-the-art segmentation models
and the real-world needs of low-resource settings. This work aims to bridge this gap by presenting the first fully annotated African multiorgan dataset for histopathology nuclei semantic segmentation acquired
with a low-precision microscope. We also evaluate state-of-the-art segmentation models, including spectral feature extraction encoder and vision transformer-based models, and stain normalization techniques for
color normalization of Hematoxylin and Eosin-stained histopathology slides. Our results provide important insights for future research on nuclei histopathology segmentation with low-resource data.

# Project 
* Datasets Used : Crynosef - TNBC - Monuseg and AMONuSeg
* Models Baselines: UNET - YNET (CNN Blocks) - YNET (FFC Blocks) - DIANet (Using DINO V1)
* Proposed Model: FD-Net 

# Proposed Model ED

# Results


# Requierement

## References
[1] A. Farshad, et al. Y-net: A spatiospectral dual-encoder network for medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 582–592. Springer, 2022.
[2] Y. Yeganehet al. Transformers pay attention to convolutions leveraging emerging properties of vits by dual attention-image network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2304–2315, 2023.
[3] J. Boschmanet al. The utility of color normalization for ai-based diagnosis of hematoxylin and eosin-stained pathology images. The Journal of Pathology, 256(1):15–24, 2022.
[4] O. Ronneberger et al. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
[5] V.  Badrinarayanan, Alex Kendall et al.A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.  
[6]Z. He, Mathias Unberathet al.. Transnuseg: A lightweight multi-task transformer for nuclei segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 206–215. Springer, 2023. 

