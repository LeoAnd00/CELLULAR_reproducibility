

#Make cell type labels
#ScType: https://github.com/IanevskiAleksandr/sc-type/tree/master 


library("tidyverse")

# Fetch cmd arguments
args <- commandArgs(trailingOnly = TRUE)
file_name <- args[1]
res <- as.double(args[2])

norm_counts <- read.csv(paste(file_name, ".csv", sep = ""), row.names = 1, header= TRUE)

# load libraries
lapply(c("dplyr","Seurat","HGNChelper"), library, character.only = T)

pbmc <- CreateSeuratObject(counts = t(norm_counts))

pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 4000)

# scale and run PCA
pbmc <- ScaleData(pbmc, features = rownames(pbmc)) # https://github.com/satijalab/seurat/issues/1166
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# cluster
pbmc <- FindNeighbors(pbmc, dims = 1:20)
pbmc <- FindClusters(pbmc, resolution = res)
pbmc <- RunUMAP(pbmc, dims = 1:20)

# load gene set preparation function
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
# load cell type annotation function
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

# DB file
db_ = "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx";
tissue = "Immune system" # e.g. Immune system,Pancreas,Liver,Eye,Kidney,Brain,Lung,Adrenal,Heart,Intestine,Muscle,Placenta,Spleen,Stomach,Thymus 

# prepare gene sets
gs_list = gene_sets_prepare(db_, tissue)

# get cell-type by cell matrix
es.max = sctype_score(scRNAseqData = pbmc[["RNA"]]@scale.data, scaled = TRUE, 
                      gs = gs_list$gs_positive, gs2 = gs_list$gs_negative) 

# NOTE: scRNAseqData parameter should correspond to your input scRNA-seq matrix. 
# In case Seurat is used, it is either pbmc[["RNA"]]@scale.data (default), pbmc[["SCT"]]@scale.data, in case sctransform is used for normalization,
# or pbmc[["integrated"]]@scale.data, in case a joint analysis of multiple single-cell datasets is performed.

# merge by cluster
cL_resutls = do.call("rbind", lapply(unique(pbmc@meta.data$seurat_clusters), function(cl){
  es.max.cl = sort(rowSums(es.max[ ,rownames(pbmc@meta.data[pbmc@meta.data$seurat_clusters==cl, ])]), decreasing = !0)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(pbmc@meta.data$seurat_clusters==cl)), 10)
}))
sctype_scores = cL_resutls %>% group_by(cluster) %>% top_n(n = 1, wt = scores)  

# set low-confident (low ScType score) clusters to "unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] = "Unknown"

# Add to metadata
pbmc@meta.data$customclassif = ""
for(j in unique(sctype_scores$cluster)){
  cl_type = sctype_scores[sctype_scores$cluster==j,]; 
  pbmc@meta.data$customclassif[pbmc@meta.data$seurat_clusters == j] = as.character(cl_type$type[1])
}

# Download labels
labels <- pbmc@meta.data$customclassif
write.csv(labels,paste(file_name, "_labels.txt", sep = ""), row.names = FALSE)

