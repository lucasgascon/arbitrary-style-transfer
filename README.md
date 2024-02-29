# Real-time Arbitrary Style Transfer with Adaptive Instance Normalization

This repository contains the implementation of a real-time arbitrary style transfer using Adaptive Instance Normalization (AdaIN). This project was developed as part of the Generative Modelling for Images class at MVA (Mathématiques, Vision, Apprentissage) program at École Normale Supérieure Paris-Saclay. The course was taught by Arthur Leclaire and Bruno Galerne.

## Team Members
- Hippolyte Pilchen ([forename.lastname@polytechnique.edu](mailto:forename.lastname@polytechnique.edu))
- Lucas Gascon

## Introduction
Arbitrary style transfer aims to apply the artistic style of one image (the style image) to another image (the content image) while preserving the content of the latter. This project explores real-time style transfer, allowing for interactive applications.

## Methodology
The core technique used in this project is Adaptive Instance Normalization (AdaIN). AdaIN aligns the mean and variance of the content features with those of the style features, effectively transferring the style to the content image. We implemented this technique using PyTorch, leveraging its capabilities for efficient computation on GPUs.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/your-username/arbitrary-style-transfer.git
   ```
2. Navigate to the project directory:
   ```
   cd arbitrary-style-transfer
   ```
3. Run the main script with your desired content and style images:
   ```
   python main.py --content content_image.jpg --style style_image.jpg
   ```
4. Optionally, adjust the hyperparameters such as style weight, content weight, and the number of iterations to achieve different style transfer effects.

## Results
The results of our experiments demonstrate the effectiveness of real-time arbitrary style transfer using AdaIN. We provide sample outputs in the `results` directory for reference.

## Future Work
- Optimization for even faster real-time performance.
- Integration with web or mobile applications for interactive style transfer experiences.
- Exploration of alternative style transfer techniques for comparison and improvement.

## Acknowledgements
We would like to thank our instructors, Arthur Leclaire and Bruno Galerne, for their guidance and support throughout this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
