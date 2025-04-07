# Human Face Colorization

## Overview
This project focuses on human face potrait colorization using deep learning techniques. The core functionalities of the baseline model are taken from [M. Berkay's Image colorization](https://github.com/mberkay0/image-colorization) For further information look at the report provided as PDF.


## Dataset
We use publicly available datasets:
- [1 Million Fake Faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces)
- [Face Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/face-colorization.git
cd face-colorization
```
Because we had no access to local GPU compute ressources we ran most of the code inside of a jupyter notebooks on [Google Colab](https://colab.research.google.com/).
Either way, we created an environment.yml with conda if you want to run this code locally. This can be done by running

```bash
# Clone the repository
conda env create --file environment.yml
conda activate my-project
```

Then you can just open the demo.ipynb notebook to reproduce the experiments or colorize your own images.
There is also code that can be found in the src folder to preprocess images. 


## Contributors
- [Elias Huber](https://github.com/yourusername)
- [Sam Rouppe Van der Voort](https://github.com/teammateusername)
- [Oliver Sange](https://github.com/teammateusername)

We also want to note that we used LLMs for creating Numpy style docstrings on some functions in the code.