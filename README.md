# Res2Unet
A new FCNN for biomedical image segmentation. We implemented an efficient architecture based on the Heun's method to improve the network perfomance withoud needing of more layers.
![Res2Unet architecture](https://github.com/ojedaa/Res2Unet/blob/main/Arq1_2.png)

Res2Unet is based on the U-Net [https://doi.org/10.1007/978-3-319-24574-4_28] and ResUnet [doi: 10.1109/LGRS.2018.2802944] original architectures. 

With this architecture we improved the perfomance of T. cruzi parasite segmentation in compare with the U-Net and ResUnet models. We obtain the best performance using the Active Contour Loss Function [doi: https://doi.org/10.1109/CVPR.2019.01190] and the residual functions presented on [https://doi.org/10.1007/978-3-319-46493-0_38].


Publication:
Ojeda-Pat, A., Martin-Gonzalez, A., Brito-Loeza, C, & Ruiz-Piña, H. (2020). Res2Unet: A Fully Convolutional Neural Network for Trypanosoma cruzi parasite segmentation. Artificial Intelligence in Medicine, XX(XX), xx-xx.
