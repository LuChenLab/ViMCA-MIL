<img width="432" height="14" alt="image" src="https://github.com/user-attachments/assets/fc8207c3-564e-42e8-9b6c-15b082f1dd19" /># ViMCA-MIL

**Vi**ral **M**utation–**C**linical **A**ssociation analysis based on a gated-attention **M**ultiple **I**nstance **L**earning model
<img width="2126" height="616" alt="image" src="https://github.com/user-attachments/assets/09cec173-c381-4bc5-80ae-653dba06e56b" />
</br>

## What is ViMCA-MIL?

ViMCA-MIL is an integrated analysis framework that screens for SARS-CoV-2 mutations
which simultaneously confer viral evolutionary advantages and impact host clinical
phenotypes. It integrates large-scale genomic surveillance data with matched
virus–patient clinical records, and prioritizes candidate mutations through a
gated-attention multiple-instance learning (MIL) model.

Briefly, the framework consists of four steps:

1. **Evolutionary analysis.** Variant and mutation frequency trajectories are
   characterized across major SARS-CoV-2 variants using ~15.5 million genomic
   sequences from GISAID (Dec 2019 – Jan 2026) together with locally sequenced
   genomes from West China Hospital, Sichuan University.
   Recurrent mutations are clustered into evolutionary clusters to distinguish
   *stable* from *sporadic* mutations.
2. **Fitness and transmission analysis.** Relative viral fitness (R/R<sub>A</sub>)
   is estimated with the [PyR0](https://github.com/broadinstitute/pyro) workflow on
   the UShER phylogeny, and transmission selection coefficients are inferred from
   regional genomic surveillance data following
   [Lee *et al.*](https://github.com/bartonlab/paper-SARS-CoV-2-inference).
   EVEscape-derived fitness effects are additionally computed per protein.
3. **Clinical phenotype analysis.** For a cohort of 490 patients with matched viral
   genomes and clinical records (78 blood routine, blood chemistry, immune cell and
   cytokine/chemokine features), features are normalized (inverse normal
   transformation), adjusted for age, sex and a Charlson comorbidity index (CCI)
   weighted clinical burden score, and filtered by temporal correlation with the
   course of the pandemic, yielding 38 temporally correlated traits.
4. **Gated-attention MIL modeling.** Each patient is treated as a *bag* of the
   mutations detected in the matched viral genome; each mutation is an *instance*
   represented by 15 mutation-level features. The model predicts each patient-level
   clinical feature value from the attention-weighted aggregation of its mutations,
   and the attention scores are used to prioritize mutations without requiring
   mutation-level labels. Robustness is assessed with 100 independent resampling
   runs followed by 10 full-data refittings to obtain mean attention scores.

</br>

## Contents

[Repository structure](#repository-structure)  
[Requirements](#requirements)  
[Input](#input)  
[Output](#output)  
[Tutorial](#tutorial)  
[Mutation features](#mutation-features)  
[Contact](#contact)  
[Citation](#citation)

</br>

## Repository structure

```
ViMCA-MIL/
├── README.md
├── script/
│   ├── 01Evolutionary_analysis_variant_mutations_v2.Rmd   # Variant/mutation evolutionary dynamics
│   ├── 02Fitness_analysis_v2.Rmd                          # Relative fitness & transmission selection
│   ├── 03Clinical_phenotype_analysis_v2.Rmd               # Clinical feature preprocessing & temporal analysis
│   ├── MIL_bootstrap_multi_trait_260714.py                # Gated-attention MIL model (single trait per run)
│   ├── run_MIL.sh                                         # Batch launcher for all traits + summary merging
│   └── count_nm_sm_by_sequence_gene.py                    # Per-sequence/per-gene Nm & Sm counting (dN/dS)
└── data/
    ├── Chengdu_data/                                      # Local cohort lineage metadata
    ├── SARS_CoV_2_reference/                              # Reference genome & gene structure
    ├── clinical_phenotype_analysis/                       # Clinical phenotypes & MIL inputs
    │   └── MIL/prepeare/                                  # geno_feature.csv, phenotype matrix, Cov2Var annotations
    ├── GISAID_260111/                                     # Derived GISAID results (mutation frequencies, clusters,
    │                                                      #   selection coefficients, dN/dS annotations)
    └── relative_fitness/pyr0/                             # PyR0 relative fitness results
```

</br>

## Requirements

### Python

| Software | Version |
| --- | --- |
| Python | 3.9.25 |
| torch | 2.8.0 (CUDA 12.8) |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.13.1 |
| tqdm | 4.67.1 |
| biopython | 1.85 |

```bash
conda create -n MIL python=3.9
conda activate MIL
pip install torch==2.8.0 numpy==1.26.4 pandas==2.3.3 \
    scikit-learn==1.6.1 scipy==1.13.1 tqdm==4.67.1 biopython==1.85
```

The MIL model automatically falls back to CPU when no CUDA device is available.

### R

Developed with `R 4.5.3`. Packages used by the Rmd scripts:

| Category | Packages (version) |
| --- | --- |
| Data wrangling | data.table (1.18.2.1), dplyr (1.2.1), tidyr (1.3.2), purrr (1.2.2), stringr (1.6.0), magrittr (2.0.5), readxl (1.4.5), openxlsx (4.2.8.1), broom (1.0.12), lubridate (1.9.5), scales (1.4.0) |
| Statistics & modeling | caret (7.0.1), bestNormalize (1.9.2), binom (1.1.1.1), parallel (4.5.3) |
| Visualization | ggplot2 (4.0.3), ggpubr (0.6.3), ggrepel (0.9.8), patchwork (1.3.2), cowplot (1.2.0), ggthemes (5.2.0), ggsci (5.0.0), RColorBrewer (1.1.3), circlize (0.4.18), ComplexHeatmap (2.26.1) |
| Parallel computing | future (1.70.0), future.apply (1.20.2) |
| Bioconductor | Biostrings (2.78.0)  |
| Sequence handling | seqinr (4.2.36) |

External tools/workflows invoked by the analysis notebooks:

- [PyR0](https://github.com/broadinstitute/pyro) — relative fitness estimation
  (`preprocess_usher.py`, `mutrans.py` called in `02Fitness_analysis_v2.Rmd`)
- [EVcouplings](https://github.com/debbiemarkslab/EVcouplings) — MSA construction
  for EVEscape fitness-effect scores (`evcouplings_runcfg` called in
  `03Clinical_phenotype_analysis_v2.Rmd`)
- [Transmission selection coefficients inference](https://github.com/bartonlab/paper-SARS-CoV-2-inference) — transmission
  selection coefficients

Upstream sequencing preprocessing of the local cohort (not part of this
repository) used [fastp](https://github.com/OpenGene/fastp) (v0.23.2),
[bowtie2](https://github.com/BenLangmead/bowtie2) (v2.4.4) and
[Nextclade](https://clades.nextstrain.org/) for mutation calling and lineage
assignment.

</br>

## Input

All processed input data required by the MIL model and the downstream analyses
are provided in `data/` of this repository. Patient identifiers are anonymized.

#### 1. Mutation feature matrix — `data/clinical_phenotype_analysis/MIL/prepeare/geno_feature.csv`

A long-format data frame in which each row is one mutation detected in one patient,
containing at least the following columns:

| Column | Description |
| --- | --- |
| `sample_id` | Patient/sample identifier (defines the MIL *bag*) |
| `mutation_id` | Mutation identifier (e.g. `Spike:D614G`) |
| 15 feature columns | See [Mutation features](#mutation-features) |

#### 2. Clinical phenotype matrix — `data/clinical_phenotype_analysis/MIL/prepeare/nor_pheno_by_Age_Gender_CCI.csv`

A data frame with one row per patient (`sample_id`) and one column per clinical
feature. Values are normalized, covariate-adjusted (age, sex, CCI) and
standardized, as described in `03Clinical_phenotype_analysis_v2.Rmd`.

#### 3. Trait list — `data/clinical_phenotype_analysis/MIL/traits.txt`

A plain-text file listing one clinical trait (column name of the phenotype matrix)
per line. `run_MIL.sh` launches one MIL job per trait.

#### 4. Multiple sequence alignment (optional, dN/dS analysis)

A FASTA (optionally gzipped) multiple sequence alignment of SARS-CoV-2 genomes
(e.g. the GISAID MSA), plus the gene/feature annotation table
(`data/GISAID_260111/dN_dS/sarscov2_features.tsv`), used by
`count_nm_sm_by_sequence_gene.py`.

#### Data not included

Due to size and data-sharing agreements, the following raw inputs are not
included, but the corresponding derived results are provided in `data/`:

- GISAID sequence metadata (~15.5 million records) — download from
  [GISAID](https://www.gisaid.org/) (registration required).
- The GISAID multiple sequence alignment — available from GISAID.
- The UShER phylogenetic tree for PyR0 — download from the
  [UCSC Genome Browser](http://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2).

</br>

## Output

Running `run_MIL.sh` produces, for each trait, under `OUT_DIR`:

- `predictions/` — observed vs. predicted values for each resampling run
- `attention_each_run/` — mutation attention scores per run
- `model_weights/` — trained model weights
- `summary_each_trait/Summary_*.csv` — per-trait performance summary
  (mean Pearson *r* over the 100 resampling runs, etc.)
- `Summary_All_Traits.csv` — merged summary across all traits
- `logs/` — per-trait log files

The final mutation–clinical feature association is quantified by the **mean
attention score** of each mutation for each clinical feature obtained from the 10
full-data refittings. Mutations with attention scores above the mean across all
evaluated mutations are defined as high-attention mutations for that feature.

</br>

## Tutorial

The main executable deliverable of this repository is the gated-attention MIL
pipeline. The R notebooks (`01`–`03`) are provided as reference implementations
of the upstream analyses described above; their derived results are already
included in `data/`, so the MIL model can be run directly.

### Gated-attention MIL modeling

`script/MIL_bootstrap_multi_trait_260714.py` + `script/run_MIL.sh`

Single-trait usage (run from the `script/` directory so that the default
`--geno`/`--pheno` paths resolve to `../data/`):

```bash
cd script
CUDA_VISIBLE_DEVICES=0 python MIL_bootstrap_multi_trait_260714.py \
    --trait "TNF_a" \
    --output_dir ../output/multi_traits_results \
    --n_boot 100 \
    --n_full 10 \
    --epochs 150 \
    --base_seed 42 \
    --r_threshold 0 \
    --save_phase1_attention 1 \
    --save_phase1_pred 1 \
    --save_phase2_model 1 \
    --save_phase2_attention 1 \
    --save_phase2_pred 1
```

Data paths can be overridden with `--geno` and `--pheno`.

The pipeline runs in two phases:

- **Phase 1 — stability screening**: 100 independent resampling runs; in each run
  80% of patients are used for training and 20% for validation. Models are trained
  for 150 epochs with MSE loss and the AdamW optimizer (lr = 5×10⁻⁴, weight decay
  = 0.01).
- **Phase 2 — refitting**: for traits whose mean validation Pearson correlation
  exceeds the threshold (`--r_threshold`), the model is refit 10 times on all
  matched samples with different seeds; mean mutation attention scores are
  extracted.

To run all traits in parallel (6 jobs per batch by default), run from anywhere
in the repository (paths are resolved relative to the repository root):

```bash
bash script/run_MIL.sh
```

Results are written to `output/multi_traits_results/`.

Downstream, traits whose mean validation Pearson correlation exceeds the upper
bound of the 95% confidence interval across all evaluated traits are retained for
mutation prioritization (13 traits in our study, with serum TNF-α being the most
predictable).

</br>

## Mutation features

Each mutation instance is represented by 15 features:

| Category | Features | Source |
| --- | --- | --- |
| Conservation | `phastCons`, `phyloP` (max over the affected codon, 119 coronavirus genomes) | UCSC Genome Browser |
| Fitness & transmission | `relative_fitness` (R/R<sub>A</sub>), `selection_coefficient`, `evo_idx` (EVEscape fitness effect) | PyR0 / Lee *et al.* / EVEscape |
| Protein stability | `delta_DDG_Env`, `delta_DDG_Int` (predicted ΔΔG under environmental/intracellular conditions) | Cov2Var |
| Physicochemical properties | `Molecular_weight`, `Theoretical_PI`, `Extinction_coefficients`, `Aliphatic_index`, `grand_average_of_hydropathicity` | Cov2Var |
| Functional impact | `Protein_Func` (pathogenicity), `SIFT_Probability`, `PROVEAN_Score` | Cov2Var / SIFT / PROVEAN |

</br>

## Contact

For questions about the code, please open an issue or contact
Lu Chen (luchen@scu.edu.cn) or Kepan Linghu (lhkp5457@163.com).

</br>

## Citation

If you find ViMCA-MIL useful, please cite our paper:

> Wei H.-C.\*, Linghu K.\*, Yang H.\*, Yang Q.\*, Huang X.\*, Wang Y.-H., *et al.*
> Effects of high-frequency clinical mutations on SARS-CoV-2 replication and
> virulence. *(under submission)*

</br>

