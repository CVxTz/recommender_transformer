# recommender_transformer

### Setup (GPU)
```
conda create -n py38 python=3.8
conda activate py38
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge jupyterlab
conda install -c conda-forge matplotlib
https://github.com/CVxTz/recommender_transformer
cd recommender_transformer
pip install .
```
### Docker (CPU)
```bash
docker build . -t recommender
ocker run recommender sh -c "python3.8 -m pytest"
```

### References

[BERT4Rec: Sequential Recommendation with Bidirectional
Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)