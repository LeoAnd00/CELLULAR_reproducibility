# novel_cell_type_detection
Contains all code used in this project for evaluating novel cell type prediction ability.

## How to run
In *benchmark_annotation.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train CELLULAR for each fold when leaving out every type of cell in the MacParland dataset. <br><br>
In *benchmark_annotation_Segerstolpe.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train CELLULAR for each fold when leaving out every type of cell in the Segerstolpe dataset. <br><br>
In *benchmark_annotation_Baron.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train CELLULAR for each fold when leaving out every type of cell in the Baron dataset. <br><br>
In *benchmark_annotation_Zheng68k.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train CELLULAR for each fold when leaving out every type of cell in the Zheng68k dataset. <br><br>
In *visualization/* there's code to create data for visualization of min confidence for novel and non-novel cells of each fold for each dataset and each cell type dropout event. Go to *novel_cell_type_confidence.py* and execute the command given as a comment in that file. This will create a file called *likelihood.json* where all minimum likelihoods are saved. Run *novel_cell_type_confidence.ipynb* to make figures.
