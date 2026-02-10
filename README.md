# 🧬 RAIDNC — RandomForest Assisted Discrimination of Non-Coding transcripts

RAIDNC is a command-line tool that classifies transcript sequences as **Protein-Coding** or **lncRNA** using a pre-trained Random Forest model. It takes FASTA files as input and outputs a TSV table with predictions and class probabilities for each transcript.

## 🧑‍⚕️Author

Andrea Bonomo

## 🔬 How It Works

RAIDNC uses a k-mer frequency approach combined with sequence length to build a feature representation of each transcript:

1. 🔤 **K-mer extraction** — Each nucleotide sequence is decomposed into overlapping k-mers of size 12 (e.g., `ATGCATGCATGC`). This captures local sequence composition patterns that differ between coding and non-coding transcripts.

2. 📊 **Vectorization** — The k-mer strings are transformed into a sparse count matrix using a pre-fitted vectorizer (`kmer_vectorizer.joblib`), ensuring the same vocabulary used during training.

3. ⚖️ **Normalization** — K-mer counts are normalized by the total count per sequence, converting raw counts into relative frequencies. This makes the features comparable across sequences of different lengths.

4. 📏 **Sequence length** — The length of each transcript (in nucleotides) is appended as an additional feature, since lncRNAs and protein-coding transcripts tend to differ in length distributions.

5. 🎯 **Prediction** — The combined feature matrix (normalized k-mer frequencies + sequence length) is fed into a pre-trained Random Forest classifier (`rf_model.joblib`), which outputs a predicted class and probability estimates for each transcript.

## 📦 Requirements

### 🐍 Python packages

```
pandas
numpy
biopython
scikit-learn
scipy
joblib
tqdm
```

Install all dependencies with:

```bash
pip install pandas numpy biopython scikit-learn scipy joblib tqdm
```

### 🗂️ Model files

The following pre-trained files must be present in the same directory as `RAIDNC.py` (or in a directory specified via `--model_dir`):

- `rf_model.joblib` — Trained Random Forest model
- `kmer_vectorizer.joblib` — Fitted k-mer count vectorizer

## 🚀 Usage

### 📄 Single FASTA file

Predict all transcripts in a single FASTA file:

```bash
python RAIDNC.py --fasta input.fa
```

This will create `predictions.tsv` in the current directory. To specify a custom output path:

```bash
python RAIDNC.py --fasta input.fa --output results/my_predictions.tsv
```

### 📁 Batch mode (directory of FASTA files)

Predict all FASTA files (`.fa`, `.fasta`, `.fna`) in a directory:

```bash
python RAIDNC.py --batch_dir path/to/fasta_dir/
```

Control the number of parallel threads (defaults to the number of CPU cores):

```bash
python RAIDNC.py --batch_dir path/to/fasta_dir/ --output batch_results.tsv -t 4
```

### 🔧 Using a different model directory

If your model files are stored elsewhere:

```bash
python RAIDNC.py --model_dir path/to/model/ --fasta input.fa
```

### ❓ Full help

```bash
python RAIDNC.py --help
```

## ⚙️ Command-line Options

| Option | Description | Default |
|---|---|---|
| `--fasta` | 📄 Path to a single FASTA file to predict | *(required, or use `--batch_dir`)* |
| `--batch_dir` | 📁 Path to a directory containing multiple FASTA files | *(required, or use `--fasta`)* |
| `--output` | 💾 Output TSV file path | `predictions.tsv` |
| `--threads`, `-t` | 🧵 Number of parallel threads for batch mode | Number of CPU cores |
| `--model_dir` | 🗂️ Directory containing `rf_model.joblib` and `kmer_vectorizer.joblib` | Script's own directory |
| `--help`, `-h` | ❓ Show usage information and exit | |

## 📋 Output Format

The output is a tab-separated (TSV) file with the following columns:

| Column | Description |
|---|---|
| `Transcript_ID` | 🏷️ Sequence identifier from the FASTA header |
| `Prediction` | 🎯 Predicted class: `Protein_Coding` or `lncRNA` |
| `Prob_Protein_Coding` | 📈 Probability of the transcript being protein-coding (0-1) |
| `Prob_lncRNA` | 📉 Probability of the transcript being a lncRNA (0-1) |
| `FASTA_File` | 📄 Source FASTA file name |

### 💡 Example output

```
Transcript_ID        Prediction      Prob_Protein_Coding  Prob_lncRNA  FASTA_File
NONATHT000001.1      lncRNA          0.0333               0.9667       NONCODEv5_arabidopsis.fa
NONATHT000004.1      lncRNA          0.0667               0.9333       NONCODEv5_arabidopsis.fa
NONATHT000005.1      lncRNA          0.1778               0.8222       NONCODEv5_arabidopsis.fa
NONATHT000006.1      lncRNA          0.0889               0.9111       NONCODEv5_arabidopsis.fa
```

## 🪟 Guide for Windows Users

### 1. 🐍 Install Python

Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/). During installation, make sure to check **"Add Python to PATH"**.

To verify the installation, open **Command Prompt** (or **PowerShell**) and run:

```cmd
python --version
```

### 2. 📦 Install dependencies

```cmd
pip install pandas numpy biopython scikit-learn scipy joblib tqdm
```

If you have multiple Python versions installed, you may need to use `pip3` instead of `pip`.

### 3. ▶️ Running RAIDNC

Open **Command Prompt** or **PowerShell**, navigate to the folder containing `RAIDNC.py`, and run the script. On Windows, remember to use backslashes (`\`) in file paths, or wrap paths in double quotes.

**📄 Single file:**

```cmd
python RAIDNC.py --fasta "C:\Users\YourName\Documents\sequences\input.fa"
```

**📁 Batch directory:**

```cmd
python RAIDNC.py --batch_dir "C:\Users\YourName\Documents\fasta_files" --output results.tsv -t 4
```

**💾 Custom output path:**

```cmd
python RAIDNC.py --fasta input.fa --output "C:\Users\YourName\Desktop\predictions.tsv"
```

### 4. 📝 Paths with spaces

If any folder or file name contains spaces, always wrap the entire path in double quotes:

```cmd
python RAIDNC.py --fasta "C:\My Folder\my sequences.fa"
```

### 5. 🐍 Using Anaconda/Miniconda (alternative)

If you use Anaconda or Miniconda, open the **Anaconda Prompt** and create a dedicated environment:

```cmd
conda create -n raidnc python=3.10
conda activate raidnc
conda install pandas numpy scikit-learn scipy joblib tqdm -c defaults
conda install biopython -c conda-forge
```

Then run the script from the Anaconda Prompt as shown above.

## 📜 License

