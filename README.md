# Reproducibility code | CELLULAR: CELLUlar contrastive Learning for Annotation and Representation

**Abstract**
<br>
Batch effects are a significant concern in single-cell RNA sequencing (scRNA-Seq) data analysis, where variations in the data can be attributed to factors unrelated to cell types. This can make downstream analysis a challenging task. In this study, we present a novel deep learning approach using contrastive learning and a carefully designed loss function for learning an generalizable embedding space from scRNA-Seq data. We call this model CELLULAR: CELLUlar contrastive Learning for Annotation and Representation. When benchmarked against multiple established methods for scRNA-Seq integration, CELLULAR outperforms existing methods in learning a generalizable embedding space on multiple datasets. Cell annotation was also explored as a downstream application for the learned embedding space. When compared against multiple well-established methods, CELLULAR demonstrates competitive performance with top cell classification methods in terms of accuracy, balanced accuracy, and F1 score. CELLULAR is also capable of performing novel cell type detection. These findings aim to quantify the \textit{meaningfulness} of the embedding space learned by the model by highlighting the robust performance of our learned cell representations in various applications. The model has been structured into an open-source Python package, specifically designed to simplify and streamline its usage for bioinformaticians and other scientists interested in cell representation learning.

## Structure
- **code:** Contains all code used in this project, including preprocessing, visualization, machine learning models and more.
- **data:** Contains all data used in this project, including raw and preprocessed data.

## Necessary programming languages
- Python version 3.10.5
- R version 4.3.2

## Data
Data for reproducibility can be found [here](https://doi.org/10.5281/zenodo.10959788).

## Citation
Coming soon
