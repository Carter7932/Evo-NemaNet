# Evo-NemaNet

## 1. Installation
- Use Python 3.9 to match dependency versions.
- Install packages from the project root:
  ```bash
  pip install -r requirements.txt
  ```
- Download weights: SAM ViT-H (`sam_vit_h_4b8939.pth`) from the official Segment Anything release (https://github.com/facebookresearch/segment-anything#model-checkpoints) and MambaVision tiny/small (`mambavision_tiny_1k.pth.tar`, `mambavision_small_1k.pth.tar`) from the official MambaVision releases (https://github.com/state-spaces/mamba-vision).
- The Checkpoint of Evo-NemaNet and Data of Evo-Nema Dataset can be achieved in the following link
- Best_Model: [link] (code: ____)
- I-Nema dataset: [link] (code: ____)
- Evo-Nema dataset: [link] (code: ____)
- Test Data:

## 2. Script Guide
- `SAM_NemaSeg.py`  
  Uses the proposed SAM-NemaSeg head-feature segmentation algorithm for robust, precise nematode head extraction. Reads raw images from `Test_Raw_Data` and writes segmented results to `Test_SAM` (preserving folder structure if configured).
- `Evo-NemaDataset_Construction.py`  
  Reorganizes the original I-Nema data; balances classes with augmentations (manual `Data_Purification` is external) and outputs to `Evo-Nema_Dataset/Train` and `Evo-Nema_Dataset/Val`.
- `Train.py`  
  Train a single backbone
- `Train_All.py`  
  Sequentially train a list of backbones
- `Test.py`  
  Test a single backbone
- `Test_All.py`  
  Sequentially test a list of backbones
## 3. Workspace Layout
- `Best_Model/` Trained checkpoints (EfficientNet, Swin, MambaVision, ResNet variants).
- `I-Nema_Dataset/Train`, `I-Nema_Dataset/Val`  Original cleaned dataset split.
- `Evo-Nema_Dataset/Train`, `Evo-Nema_Dataset/Val` Balanced and augmented dataset produced by `Evo-NemaDataset_Construction.py`.
- `Test_Raw_Data/` Raw test images acquired form "NemaRec"
- `Test_SAM/` Segmentation outputs from `SAM_NemaSeg.py` (created at runtime).
- `utils_model_checkpoint/` Supporting weights (SAM ViT-H, MambaVision tiny/small).
## 4. EvoNemaNet Construction
The EfficientNetV2-S was selected as the final backbone since it achieves the best classification performance among the backbones
## 5. Test Raw Data Source
Test raw data were obtained from:
Xue Qing, Yihao Wang, Xuequan Lu, Haibo Li, Xuan Wang, Hongmei Li, Xiaojun Xie. NemaRec: A deep learning-based web application for nematode image identification and ecological indices calculation. European Journal of Soil Biology, Volume 110, 2022, 103408. ISSN 1164-5563. https://doi.org/10.1016/j.ejsobi.2022.103408 (https://www.sciencedirect.com/science/article/pii/S1164556322000255). The data are cited to ensure fair evaluation.
## 6. Citations
Hatamizadeh, A., \& Kautz, J. (2025). Mambavision: A hybrid mamba-transformer vision backbone. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 25261-25270).

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... \& Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).

Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., ... \& Guo, B. (2022). Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12009-12019).

Tan, M., \& Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.

Tan, M., \& Le, Q. (2021, July). Efficientnetv2: Smaller models and faster training. In International conference on machine learning (pp. 10096-10106). PMLR.

He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
