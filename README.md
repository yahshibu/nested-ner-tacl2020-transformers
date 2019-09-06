# Implementation of Nested Named Entity Recognition

Some files are part of [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2).

## Requirements

We tested this library with the following libraries:

* Python (3.7)
* [PyTorch](https://github.com/pytorch/pytorch) (1.0.1)
* [Numpy](https://github.com/numpy/numpy) (1.16.2)
* [AdaBound](https://github.com/Luolc/AdaBound) (0.0.5)
* [StanfordNLP](https://github.com/stanfordnlp/stanfordnlp) (0.1.2) for accessing the Java Stanford CoreNLP Server (3.9.2)
* [PyTorch Pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT) (0.6.1)

## Running experiments

#### Testing this library with a sample data

1. Run the **gen_data.py** to generate the processed data files for training, and they will be placed at the "./data/" directory
   ```bash
   python gen_data.py
   ```
2. Run the **train.py** to start training
   ```bash
   python train.py
   ```

#### Reproducing our experiment on the ACE-2004 dataset

1. Put the corpus [ACE-2004](https://catalog.ldc.upenn.edu/LDC2005T09) into the "../ACE2004/" directory
2. Put [this .tgz file](http://www.statnlp.org/research/ie/code/statnlp-mentionextraction.v0.2.tgz) into the "../" and extract it
3. Run the **parse_ace2004.py** to extract sentences for training, and they will be placed at the "./data/ace2004/"
   ```bash
   python parse_ace2004.py
   ```
4. Run the **gen_data_for_ace2004.py** to prepare the processed data files for training, and they will be placed at the "./data/" directory
   ```bash
   python gen_data_for_ace2004.py
   ```
5. Run the **train.py** to start training
   ```bash
   python train.py
   ```

#### Reproducing our experiment on the ACE-2005 dataset

1. Put the corpus [ACE-2005](https://catalog.ldc.upenn.edu/LDC2006T06) into the "../ACE2005/" directory
2. Put [this .tgz file](http://www.statnlp.org/research/ie/code/statnlp-mentionextraction.v0.2.tgz) into the "../" and extract it
3. Run the **parse_ace2005.py** to extract sentences for training, and they will be placed at the "./data/ace2005/"
   ```bash
   python parse_ace2005.py
   ```
4. Run the **gen_data_for_ace2005.py** to prepare the processed data files for training, and they will be placed at the "./data/" directory
   ```bash
   python gen_data_for_ace2005.py
   ```
5. Run the **train.py** to start training
   ```bash
   python train.py
   ```

#### Reproducing our experiment on the GENIA dataset

1. Put the corpus [GENIA](http://www.geniaproject.org/genia-corpus/pos-annotation) into the "../GENIA/" directory
2. Run the **parse_genia.py** to extract sentences for training, and they will be placed at the "./data/genia/"
   ```bash
   python parse_genia.py
   ```
3. Run the **gen_data_for_genia.py** to prepare the processed data files for training, and they will be placed at the "./data/" directory
   ```bash
   python gen_data_for_genia.py
   ```
4. Run the **train.py** to start training
   ```bash
   python train.py
   ```

## Configuration

Configurations of the model and training are in **config.py**

## Citation

Please cite [our arXiv paper](https://arxiv.org/abs/1909.02250):

```
@article{shibuya2019nested,
  title={Nested Named Entity Recognition via Second-best Sequence Learning and Decoding},
  author={Shibuya, Takashi and Hovy, Eduard},
  journal={arXiv preprint arXiv:1909.02250},
  year={2019}
}
```
