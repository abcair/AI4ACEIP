### AI4ACEIP： A Computing Tool to Identify Food Peptides with High Inhibitory Activity for ACE by Merged Molecular Representation and Rich Intrinsic Sequence Information Based on Ensemble Learning Strategy 
Hypertension, a widespread chronic condition and a key contributor to cardiovascular diseases is often managed by inhibiting the angiotensin-converting enzyme (ACE). This enzyme converts angiotensin I to angiotensin II, leading to vasoconstriction and increased blood pressure. While pharmacotherapy is the standard treatment, it can cause adverse effects. Food-derived ACE-inhibiting peptides (ACEIP) present a promising alternative with fewer side effects. Our study introduces AI4ACEIP, a cutting-edge model designed to identify ACEIP in food using a two-layer stacked ensemble architecture and features from sequences, language models, and molecular data. We found that certain feature pairs, such as (AAindex1, Prot-t5_xl_bfd), (ESM2, Prot-t5_xl_uniref50), (Prot-t5_xl_uniref50, RdkitDescriptors), and (Prot-t5_xl_bfd, PubChem10M), significantly enhance ACEIP prediction performance. The PowerShap feature selection method was employed to choose 40 optimal feature and meta-model combinations, resulting in AI4ACEIP outperforming existing methods by 5-20%. This reliable prediction model, AI4ACEIP, is available for public access at https://github.com/abcair/AI4ACEIP.  
### Installation <br>
1. download the source code and model at https://github.com/abcair/AI4ACEIP.git by command `git clone https://github.com/abcair/AI4ACEIP.git` <br>
2. create a Python running environment  <br>
   >  (1) please install conda soft, see https://docs.anaconda.com/miniconda/ for how to install conda   <br>
   > (2) create a running environment by `conda create -y -n AI4ACEIP python=3.9`  <br>
   > (3) install the dependent library by `pip install -r requirements.txt` <br>
### Training a new model for your needs
1. put the negative samples and positive samples with fasta format into neg.fa and pos.fa at data directory <br>
2. running the command by `python train_run.py` <br>
### Predicting ACEIP by peptide sequences <br>
1. put your fasta format sequence data in `demo.fa` file <br>
2. run the python command by `python infer_AI4ACEIP.py --fasta_path seq.fa --save_path ./res.txt`, and the predicting results will save to `res.txt` file <br>
<br>
