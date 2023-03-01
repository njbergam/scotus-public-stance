# scotus-public-stance
 Analyzing the SCOTUS-public opinion problem using automated stance detection

# Dependencies

In order to make sure packages work, use the following command and activate the virtual environment:
```conda activate nenv```
In order to exit the virtual environment, use:
```conda deactivate```
There should also be a "requirements.txt" file

# Dataset

If you would like to use the SCOTUS-stance (SC-stance) dataset, we provide the following download [link](https://drive.google.com/file/d/1yc8x7Wzpn7J3Pu_WOZ_KRmz6wyVTHquq/view?usp=share_link). These datasets are also found in the repository. Please email the authors (njb2154@columbia.edu) with any questions!


# Major Processing Files:

### Model Training and Evaluation

[run_modeling.py]
[eval.py]

### Political Stance Analysis

[scpa.py]
[written.py]
[final.py]

### SC-stance and legal stance detection

[eval_classical.py]
[train_legal_adapter.py]


# Citation

Please use the following citation.

@inproceedings{bergam-etal-2022-legal,
    title = "Legal and Political Stance Detection of {SCOTUS} Language",
    author = "Bergam, Noah  and
      Allaway, Emily  and
      Mckeown, Kathleen",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.nllp-1.25",
    pages = "265--275",
    abstract = "We analyze publicly available US Supreme Court documents using automated stance detection. In the first phase of our work, we investigate the extent to which the Court{'}s public-facing language is political. We propose and calculate two distinct ideology metrics of SCOTUS justices using oral argument transcripts. We then compare these language-based metrics to existing social scientific measures of the ideology of the Supreme Court and the public. Through this cross-disciplinary analysis, we find that justices who are more responsive to public opinion tend to express their ideology during oral arguments. This observation provides a new kind of evidence in favor of the attitudinal change hypothesis of Supreme Court justice behavior. As a natural extension of this political stance detection, we propose the more specialized task of legal stance detection with our new dataset SC-stance, which matches written opinions to legal questions. We find competitive performance on this dataset using language adapters trained on legal documents.",
}
