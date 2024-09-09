# Preprocessing from raw data 
- The following preprocessing steps can be quite tedious. Please post issues if you cannot run the scripts.

- datasets: [Microlens](https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/)  
-- Rating file in `Files/Small subsets for experimentation`  
-- Meta files in `Per-category files`, [metadata], [image features]  

There has been an issue with the dataset site lately, 
as it automatically redirects to an updated version of the dataset. 
Keep pressing `ESC` to stop the redirecting action.

## Required Pretrained Models:

1. **Text Features:** [sentence-transformers](https://huggingface.co/sentence-transformers)/[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
2. **Image Features:** [ViT/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
3. **Video Features:** [video-swin-transformer](https://huggingface.co/Tonic/video-swin-transformer)

## Step by step
1. Performing 5-core filtering, re-indexing - `run 0rating2inter.ipynb`
2. Train/valid/test data splitting - `run 1spliting.ipynb`
3. Reindexing feature IDs with generated IDs in step 1 - `run 2reindex-feat.ipynb`
4. Encoding text/image features - `run 3feat-encoder.ipynb`
5. Filling your data description file `*.yaml` under `src/configs/dataset` with the generated file names `*.inter`, `*-feat.npy`, etc.
6. Specifying your evaluated dataset by cmd: `python -d sports -m BM3`.


## DualGNN requires additional operation to generate the u-u graph
1. Run `dualgnn-gen-u-u-matrix.py` on a dataset `baby`:  
`python dualgnn-gen-u-u-matrix.py -d baby`
2. The generated u-u graph should be located in the same dir as the dataset.
