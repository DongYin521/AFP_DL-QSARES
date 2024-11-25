# Accelerating *de novo* design of antifungal peptides using pre-trained protein language models by Kedong Yin, Ruifang Li et al.



This repository consists of two parts: 1. A method for generating candidate antifungal peptides (AFPs) sequences based on recombining dominant amino acids (dipeptide components). 2. A method for predicting AFP activity based on the ESM-2 pre-trained model (ESM2-AFPpred). The combination of these two methods can accelerate the *de novo* design of AFPs.

It is based on the article "**Deep learning combined with quantitative structure - activity relationship accelerates *de novo* design of antifungal peptides**" by Kedong Yin, Ruifang Li, and others. This repository includes Python code, and weight files for ESM2-AFPpred.

## 1.Download pre trained models and cache them locally

Please download the pre trained model cache used in this study from Hugging Face ([facebook/esm2_t30_150M_UR50D at main](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main)) and store it in .\models--facebook--esm2_t30_150M_UR50D\. 

The cache files that need to be downloaded include:

```
config.json
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
vocab.txt
```

## 2.Generation of candidate antifungal peptides

```python
python3 c_AFPs-Gen.py
```

The dominant amino acids need to be set at n1-ni. Please refer to the main text for the calculation of dominant amino acids. Use MySQL database to store the generated candidate antifungal peptide sequences and their corresponding physicochemical properties. Users need to set parameters such as user, password, host, database, port, etc. themselves. 'readpkl.py' and 'writepkl.py' are used to read and write the weights of amino acid physicochemical properties.

## 3.Prediction of candidate antifungal peptides

```python
python3 ESM2-AFPpred.py
```

Enter the peptide sequence to be predicted in the 'data\input.csv' file, run the code, and the result will be stored in the 'data\output.csv' file.

## 4.environment

```
environment.yaml
```

All the environments and packages that this project relies on have been packaged into 'environment.yaml'.

If you need to access the dataset or other code, please contact Kedong Yin（ 2703937842@qq.com ).