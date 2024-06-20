# Reproducibility code | CELLULAR: CELLUlar contrastive Learning for Annotation and Representation

**Abstract**
<br>
Batch effects are a significant concern in single-cell RNA sequencing (scRNA-Seq) data analysis, where variations in the data can be attributed to factors unrelated to cell types. This can make downstream analysis a challenging task. In this study, a neural network model is designed utilizing contrastive learning and a novel loss function for learning an generalizable embedding space from scRNA-Seq data. When benchmarked against multiple established methods for scRNA-Seq integration, the model outperforms existing methods in learning a generalizable embedding space on multiple datasets. A downstream application that was investigated for the embedding space was cell type annotation. When compared against multiple well-established cell type classifiers, the model in this study displayed a performance competitive with top performing methods across multiple metrics, such as accuracy, balanced accuracy, and F1 score. These findings aim to quantify the ``meaningfulness'' of the embedding space learned by the model, and highlight the potential applications of these learned cellular representations.

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
