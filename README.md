### AI4ACEIP： A Computing Tool to Identify Food Peptides with High Inhibitory Activity for ACE by Merged Molecular Representation and Rich Intrinsic Sequence Information Based on Ensemble Learning Strategy 
<br>
Hypertension is a common chronic disorder and a major risk factor for cardiovascular diseases. Angiotensin-converting enzyme (ACE) converts angiotensin I to angiotensin II, causing vasoconstriction and raises blood pressure. Pharmacotherapy is the mainstay of traditional hypertension treatment leading to various negative side effects. Some food-derived peptides can suppress ACE, named ACEIP with less undesirable effects. Therefore, it is crucial to seek strong dietary ACEIP to aid in hypertension treatment. In this paper, we propose a new model called AI4ACEIP to identify ACEIP. AI4ACEIP uses a novel two-layer stacked ensemble architecture to predict ACEIP. AI4ACEIP relies on integrated view features derived from sequence, large language models, and molecular-based information. The analysis of feature combinations reveals that four selected integrated feature pairs exhibit enhancing performance for identifying ACEIP. To identify meta models with strong learning abilities, PowerShap is used to select 40 optimal feature and meta model combinations. Compared with seven state-of-the-art methods on the source and clear benchmark datasets, AI4ACEIP get significantly outperformed by 8.47% to 20.65% and 5.49% to 14.42% for Matthew’s correlation coefficient. In brief, AI4ACEIP is a reliable model for ACEIP prediction and freely available at https://github.com/abcair/AI4ACEIP.
<br>
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
2. run the python command by `python infer_AI4ACEIP.py --fasta_path seq.fa --threshold 0.5 --save_path ./res.txt`, and the predicting results will save to `res.txt` file <br>
<br>
