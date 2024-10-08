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
![FDfinale (1)](https://github.com/user-attachments/assets/39b98370-04cf-47be-820a-ce8a248fb984)

# Results
The average Dice score of the evaluated segmentation models on the Original and pre-processed AMONuSeg dataset ![image](https://github.com/user-attachments/assets/5244adcf-f645-4660-9a68-b5ea40db8aa4)
![image](https://github.com/user-attachments/assets/a2f3ee63-414b-4ef6-b3de-dc92011cb35c)

The best performance achieved a higher average Dice score of 0.830 using both Y-Net with the original AMONuSeg and FD-Net with the StainGAN pre-processed dataset.

# Requierement

## References
[1] A. Farshad, et al. Y-net: A spatiospectral dual-encoder network for medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 582–592. Springer, 2022.
[2] Y. Yeganehet al. Transformers pay attention to convolutions leveraging emerging properties of vits by dual attention-image network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2304–2315, 2023.
[3] J. Boschmanet al. The utility of color normalization for ai-based diagnosis of hematoxylin and eosin-stained pathology images. The Journal of Pathology, 256(1):15–24, 2022.
[4] O. Ronneberger et al. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
[5] V.  Badrinarayanan, Alex Kendall et al.A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.  
[6]Z. He, Mathias Unberathet al.. Transnuseg: A lightweight multi-task transformer for nuclei segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 206–215. Springer, 2023. 
[7] Starmans, Martijn Pieter Anton, et Apostolia Tsirikoglou. « MICCAI 2024 AFRICAI Imaging Repository White Paper ». Zenodo, 14 mars 2024. https://doi.org/10.5281/zenodo.10816769.


