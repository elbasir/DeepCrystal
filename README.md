# DeepCrystal
A Deep Learning Framework for sequence-based Protein Crystallization Prediction

# Abstract

Motivation: Protein structure determination has primarily been performed using X-ray crystallography. To overcome the expensive cost, high attrition rate and series of trial-and-error settings, many in-silico methods have been developed to predict crystallization propensities of proteins based on their sequences. However, majority of these methods build their predictors by extracting features from protein sequences which is computationally expensive and can explode the feature space. <br />

We propose, DeepCrystal, a deep learning framework for sequence-based protein crystallization prediction. It uses deep learning to identify proteins which can produce diffraction quality crystals without the need to manually engineer additional biochemical and structural features from sequences. Our model is based on Convolutional Neural Networks (CNNs) which can exploit k-mer structure and interaction among sets of k-mers from the raw protein sequences. <br/>

Results: Our model surpasses previous sequence-based protein crystallization predictors in terms of accuracy, precision and recall on three independent test sets. DeepCrystal achieves an average improvement of 3.5% in accuracy, when compared to the state-of-the-art method, Crysalis II. In addition, DeepCrystal attains an average improvement of 2.6% and 3.6% for precision and recall respectively w.r.t Crysalis II on the three independent test sets. <br />

![draft](https://user-images.githubusercontent.com/393716/43463469-4d4b5660-94e1-11e8-979f-7919903f37f6.png)

# Prerequisite

 1- Python >= 3.5  <br />
 2- Keras >= 1.2.0 <br />
 3- Tensorflow >= 0.12.0 

# Run DeepCrystal on a New Test File (Fasta file)

1- Download all the basic pre-requisites before running on new test protein sequences <br />
2- Protein sequences have to be saved in a fasta format similar to following format: <br />

   .>Seq1 <br />
   MPKFYCDYCDTYLTHDSPSVRKTHCSGRKHKENVKDYYQKWMEEQAQSLIDKTTAAFQQG <br />

where >Seq1 represents the fasta id and the second line is the protein sequence. <br />

3- Download the model files ( all files *.hdf5 and files *.json) from the following link <br />

https://storage.entrydns.org/nextcloud/index.php/s/3ErNEaZiKp39x4N <br />

4- Put all the downloaded files and <file.fasta> in the same directory as Test.py <br />

5- To test your protein sequences using Test.py run the following command: <br />

python Test.py <file.fasta> <br />

6- The output will be generated in the current working directory. The name of the output file is prediction_results.csv <br />

   Seq Id, Prediction <br />


# To Train a Model (Optional)

By using Train.py you can train the model on your own data. Train.py and the fasta file have to be in the same directory. The name of the weight file, json file and the model name all have to be specified by the user and the preferred directory to save them in. <br />

The following is a simple example on how the fasta file should look like: <br />
   .>Seq1 Crystallizable <br />
    MERVAVVGVPMDLGANRRGVDMGPSALRYARLLEQLEDLGYTVEDLGDVPVSLARASRRRGRGLAYLEEIRAAALVLKERLAALPEGVFPIVLGGDHSLSMGSVAGAARGRRVGVVWVDAHADFNTPETSPSGNVHGMPLAVLSGLGHPRLTEVFRAVDPKDVVLVGVRSLDPGEKRLLKEAGVRVY <br />
   .>Seq2 Non Crystallizable <br />
    MPRSLKKGVFVDDHLLEKVLELNAKGEKRLIKTWSRRSTIVPEMVGHTIAVYNGKQHVPVYITENMVGHKLGEFAPTRTYRGHGKEAKATKKK <br />



### Model.py
This file contains the architecture of DeepCrystal model. 


