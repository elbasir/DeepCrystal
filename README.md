# DeepCrystal
A Deep Learning Framework for sequence-based Protein Crystallization Prediction

# Motivation

Protein structure determination has primarily been performed using X-ray crystallography. To overcome the expensive cost, high attrition rate and series of trial-and-error settings, many in-silico methods have been developed to predict crystallization propensities of proteins based on their sequences. However, majority of these methods build their predictors by extracting features from protein sequences which is computationally expensive and can explode the feature space.

# Getting Started

These instructions will get you a copy of the project and how to test it or run it.

1- Download all the basic prerequisite before trying to test protein sequences <br />
2- Protein sequences have to be saved in a fasta format similar to following format: <br />

   .>Seq1 <br />
   MPKFYCDYCDTYLTHDSPSVRKTHCSGRKHKENVKDYYQKWMEEQAQSLIDKTTAAFQQG <br />

where >Seq1 represents the fasta id and the second line is the protein sequence. <br />

3- Download the model files ( all files *.hdf5 and files *.json) from the following link <br />

https://storage.entrydns.org/nextcloud/index.php/s/3ErNEaZiKp39x4N <br />

4- Put all the files that downloaded from the above link in the same directory + Test.py + "your fasta_file.fasta" <br />

5- To test your protein sequences using Test.py do the following command: <br />

python Test.py "your_fasta_file.fasta" <br />

6- The output will be generated in the same directory where model files, Test.py, your fasta_file.fasta are and the name of the output file is prediction_results.csv <br />


# Prerequisite

Things you must have installed before you start working with the project:

 1- Python 3.5 or python 3.6 <br />
 2- Keras 1.2.0 or latest <br />
 3- Tensorflow 0.12.0 or latest

# Train your data
By using Train.py you can train your own data. Train.py and the fasta file have to be in the same directory. The name of the weight file, json file and the model name all have to be specified by the user and the preferred directory to save them in. <br />

The following is a simple example on how the fasta file should look like: <br />
   .>Seq1 Crystallizable <br />
    MERVAVVGVPMDLGANRRGVDMGPSALRYARLLEQLEDLGYTVEDLGDVPVSLARASRRRGRGLAYLEEIRAAALVLKERLAALPEGVFPIVLGGDHSLSMGSVAGAARGRRVGVVWVDAHADFNTPETSPSGNVHGMPLAVLSGLGHPRLTEVFRAVDPKDVVLVGVRSLDPGEKRLLKEAGVRVY <br />
   .>Seq2 Non Crystallizable <br />
    MPRSLKKGVFVDDHLLEKVLELNAKGEKRLIKTWSRRSTIVPEMVGHTIAVYNGKQHVPVYITENMVGHKLGEFAPTRTYRGHGKEAKATKKK <br />

### Model.py
It is the architecture of our model that if someone wants to check it


