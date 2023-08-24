# 'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges

Repository for the paper **_'What are you referring to?'_ Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges** accepted at SIGDIAL'23. 

The `data/` folder has the results of running each model with the SIMMC 2.0 devtest data, as well as the reported analysed results from the paper. We also include a couple of other model outputs not reported.
- If you want to try with other models, you need to generate a similar file that can be used with our script.
- If you want to re-run each of the model outputs, then you have to access each paper's original repository and run their model on the data to obtain these output files.

We also provide a Jupyter Notebook script if preferred, see `run_experiments.ipynb`

Some values may differ than those from the paper, as we have improved the tagging algorithm and fixed some bugs in the code. The results are very similar, somewhat lower across all models by a few decimals. Original results are in the respective `data/` model folders, as `reported_results.csv`.


## Set up

We recommend a conda environment, but the script to run the analysis is simple enough to run on its own.

```bash
# Conda
conda create -n ce_experiments python=3.7 numpy tqdm
conda activate ce_experiments

# OR with Python 3.7+
pip install -r requirements.txt
```

**Important**: you also need the [original SIMMC2 repository](https://github.com/facebookresearch/simmc2) to be 
available as a sibling folder to this repository, as we use the original dataset 
to extract the Clarification Exchanges and its metadata.

Below is an example, but refer to their GitHub as they may have changed how to download the data.
```bash
cd ..
git lfs install
git clone https://github.com/facebookresearch/simmc2
```

The final folder structure should look like this:

```bash
# <parent_folder>
├── simmc2/
│   ├── data/
│   │   ├── simmc2_scene_jsons_dstc10_public/
│   │   │   └── ... # scene files
│   │   ├── fashion_prefab_metadata_all.json
│   │   ├── furniture_prefab_metadata_all.json
│   │   └── simmc2_dials_dstc10_devtest.json
│   └── model
│       └── mm_dst
│           ├── ...
│           └── utils
│               ├── ...
│               └── evaluate_dst.py     # important script for evaluation
└── what-are-you-referring-to/
    ├── data/       # outputs, results
    ├── src/        # code
    ├── README.md
    ├── ...
    └── run_experiments.py

```

## Run Experiments

```bash
python run_experiments.py
```

## Cite

Bibtex:

```
@inproceedings{chiyah-garcia2023sigdial,
    title={'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges},
    author={Chiyah-Garcia, Javier and Suglia, Alessandro and Eshghi, Arash and Hastie, Helen},
    booktitle={Proceedings of the 24rd Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL'23)},
    year={2023},
    month={sep},
}
```
