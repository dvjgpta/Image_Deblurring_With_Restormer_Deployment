Restormer for Image Deblurring

This project implements Restormer, a transformer-based architecture designed for high-quality image deblurring and other restoration tasks.
The model was trained and evaluated on the GoPro dataset, a widely recognized benchmark for motion deblurring, demonstrating the ability to recover fine details and remove complex blur patterns effectively.

Overview

Traditional CNNs excel at capturing local patterns but fail to model long-range dependencies effectively.Restormer addresses this limitation using a multi-Dconv head transposed self-attention (MDTA) mechanism, enabling efficient global context modeling while maintaining computational efficiency.

This repository provides:  
1.A PyTorch implementation of Restormer.  
2.A simple web interface (HTML + JavaScript) to visualize deblurring results interactively.

The project is hosted live on [Hugging Face Spaces](https://huggingface.co/spaces/dvjgpta/Image_Deblurring_Demo) where you can try out the deblurring model directly through an interactive web interface.

View the Original Paper on [Restormer](https://arxiv.org/abs/2111.09881) 


