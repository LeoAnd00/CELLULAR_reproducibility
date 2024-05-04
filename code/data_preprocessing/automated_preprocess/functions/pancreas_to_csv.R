

# Convert to .csv from .rds

library("tidyverse")
library(("Seurat"))

path <- '../../../data/raw/pancreas_human/'
data_name <- 'pancreas_human.csv'
meta_name <- 'pancreas_human_meta.csv'

pancreas.data <- readRDS(file = paste0(path, "pancreas_1/pancreas_expression_matrix.rds"))
metadata <- readRDS(file =  paste0(path,"pancreas_1/pancreas_metadata.rds"))

#write to file
write.csv(x = pancreas.data, file = paste0(path, data_name), quote = FALSE)
write.csv(x = metadata, file = paste0(path, meta_name), quote = FALSE)

