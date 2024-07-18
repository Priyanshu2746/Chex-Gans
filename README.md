# Chex-GANS


Chex-GANS is a project aimed at generating chest X-ray images using Generative Adversarial Networks (GANs). This repository contains implementations of three different GAN architectures for this task: Vanilla GANs, Wasserstein GANs (WGANs), and Deep Convolutional GANs (DCGANs).

## Overview

Chest X-ray images play a crucial role in diagnosing various pulmonary diseases. However, acquiring a large dataset of such images for training machine learning models can be challenging due to privacy and ethical concerns. Generative models like GANs offer a promising solution by generating synthetic images that closely resemble real chest X-rays.

## GAN Architectures used

### Vanilla GANs

Vanilla GANs, also known as standard GANs, consist of a generator and a discriminator trained adversarially to generate realistic images.

### Wasserstein GANs (WGANs)

Wasserstein GANs introduce a different loss function, namely the Wasserstein distance, which provides more stable training and better convergence properties compared to standard GANs.

### Deep Convolutional GANs (DCGANs)


DCGANs utilize deep convolutional networks for both the generator and discriminator, enabling the generation of higher resolution and more realistic images.

## Notebooks

This repository contains Jupyter notebooks for each of the three GAN architectures:

- `Vanilla_GAN.ipynb`
- `WGAN.ipynb`
- `DCGAN.ipynb`

These notebooks provide step-by-step implementations of the respective GAN architectures along with training and evaluation procedures.

## Dataset Used
The dataset was collected from the VinBigDataset . It serves as a pivotal resource in medical imaging, focusing on chest X-ray interpretation.
It aims to automate the localization and classification of thoracic abnormalities, bridging diagnostic accuracy gaps between urban and rural healthcare settings.
- Total Scans: 18,000 scans encompassing diverse chest X-rays.
- Training Set: 15,000 labelled scans for model training, annotated by seasoned radiologists and the remaining 3000 for testing.
The dataset used for this application has 4000 images containing the diseases and 4000 images having no disease in them.

## Results

## DCGANs

<table>
  <tr>
    <td><img src="https://github.com/k-Rohit/Chex-GANS/assets/93335681/3f97cca3-be2a-48ff-aef9-d7eba24fad7b"></td>
  </tr>
</table>


## Wasserstein GANs (WGANs)

<table>
  <tr>
    <td><img src="https://github.com/k-Rohit/Chex-GANS/assets/93335681/60e6d2f5-b888-49eb-8b76-c25aa43e03b9">  </td>
</table>

## Vanilla GANs
<table>
  <tr>
    <td><img src="https://github.com/k-Rohit/Chex-GANS/assets/93335681/f1287c2d-7806-4b60-96c6-62919e0622c6">  </td>
  </tr>
</table>


### Comparision Metrics

#### Peak to Signal Noise Ratio
- PSNR (Peak Signal-to-Noise Ratio)
- PSNR is expressed in decibels (dB) and is typically used to measure the quality of reconstruction or the similarity between two images. It compares the similarity between an original image and a corrupted/reconstructed version:

- Higher PSNR values indicate better quality. In image compression, a PSNR of 30 dB or higher is often perceived as good quality. However, the acceptability of a PSNR value can vary depending on the application and the viewer's perception.
- Lower PSNR values indicate poorer reconstruction quality.
- For generative models like GANs, which often deal with more complex image distributions or detailed textural features, achieving high PSNR can be challenging. The PSNR values you've reported (ranging from about 11.65 dB to 12.87 dB) are relatively low for traditional image processing standards but are not uncommon in scenarios involving complex image generation tasks. In these contexts, even small improvements in PSNR can be significant.

#### Mean Squared Error
- MSE measures the average squared difference between the original and reconstructed images. It provides a clear numerical indication of error quantity:

- Lower MSE values are better, indicating less error between the original and the reconstructed images.
- Higher MSE values indicate more error, suggesting a greater divergence from the original image.
- The MSE values you've listed (ranging from 0.06 to 0.08) suggest that there is still a noticeable average difference between the generated and original images. The specific acceptability of these MSE values can vary depending on the specific application and the quality requirements.


#### Structural Similarity Index (SSIM)
- SSIM quantifies the similarity between two images, considering luminance, contrast, and structure, thus providing a comprehensive measure of image quality:
- Higher SSIM values indicate greater similarity between the original and the reconstructed images, implying less perceptual difference.
- Lower SSIM values suggest more dissimilarity between the original and the reconstructed images, indicating a larger perceptual gap.
- The SSIM values provided (ranging from 0.90 to 0.95) indicate a relatively high degree of similarity between the generated and original images. However, the acceptability of these values can vary depending on the specific application and quality requirements.

<table>
  <tr>
    <td><img src="https://github.com/k-Rohit/Chex-GANS/assets/93335681/42e99747-bf5c-4cfc-9549-84d2b20ff0e5"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/k-Rohit/Chex-GANS/assets/93335681/80bf59f5-f16d-4cbe-bb83-a2e6e4054f24"></td></td>
</table>
<td>

<table>
  <tr>
     <td><img src="https://github.com/k-Rohit/Chex-GANS/assets/93335681/ac354e69-0f8a-46ef-93e8-a975c6ddd52a"></td>
  </tr>
</table>

Clone this repository:

```bash
git clone https://github.com/your_username/Chex-GANS.git
cd Chex-GANS
