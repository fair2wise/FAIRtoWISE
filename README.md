# project_template

This template includes common configuration and settings for ALS Computing projects.

<!-- TREE START -->
<pre>
.
├── Dockerfile
├── README.md
├── _tests
│   └── test_example.py
├── mkdocs
│   ├── docs
│   │   ├── about.md
│   │   ├── assets
│   │   │   ├── als_style.css
│   │   │   └── images
│   │   │       ├── doe_logo.png
│   │   │       └── lbl_logo.png
│   │   ├── index.md
│   │   └── test.md
│   ├── mkdocs.yml
│   └── overrides
│       ├── assets
│       │   └── images
│       │       └── favicon.png
│       └── main.html
├── polymer_papers
│   ├── 10.1002adfm.201002014.pdf
│   ├── 10.1002adfm.201301121.pdf
│   ├── 10.1002adfm.201304216.pdf
│   ├── 10.1002adfm.201801874.pdf
│   ├── 10.1002adfm.201802895.pdf
│   ├── 10.1002adfm.201806262.pdf
│   ├── 10.1002adfm.201806977.pdf
│   ├── 10.1002adfm.201902238.pdf
│   ├── 10.1002adfm.201902478.pdf
│   ├── 10.1002adfm.201906855.pdf
│   ├── 10.1002adfm.202000489.pdf
│   ├── 10.1002adfm.202008699.pdf
│   ├── 10.1002adfm.202102522.pdf
│   ├── 10.1002adfm.202105304.pdf
│   ├── 10.1002adfm.202109271.pdf
│   ├── 10.1002adfm.202112511.pdf
│   ├── 10.1002adfm.202201150.pdf
│   ├── 10.1002adfm.202305611.pdf
│   ├── 10.1002adma.201102421.pdf
│   ├── 10.1002adma.201405913.pdf
│   ├── 10.1002adma.201505435.pdf
│   ├── 10.1002adma.201604603.pdf
│   ├── 10.1002adma.201606574.pdf
│   ├── 10.1002adma.201700144.pdf
│   ├── 10.1002adma.201703777.pdf
│   ├── 10.1002adma.201704713.pdf
│   ├── 10.1002adma.201705243.pdf
│   ├── 10.1002adma.201705485.pdf
│   ├── 10.1002adma.201801501.pdf
│   ├── 10.1002adma.201803045.pdf
│   ├── 10.1002adma.201806660.pdf
│   ├── 10.1002adma.201808279.pdf
│   ├── 10.1002adma.201902899.pdf
│   ├── 10.1002adma.202002784.pdf
│   ├── 10.1002adma.202005897.pdf
│   ├── 10.1002adma.202105707.pdf
│   ├── 10.1002adma.202107316.pdf
│   ├── 10.1002adma.202108317.pdf
│   ├── 10.1002adma.202108749.pdf
│   ├── 10.1002adma.202110155.pdf
│   ├── 10.1002adma.202202608.pdf
│   ├── 10.1002adma.202203379.pdf
│   ├── 10.1002adma.202205926.pdf
│   ├── 10.1002adma.202207020.pdf
│   ├── 10.1002adom.202300776.pdf
│   ├── 10.1002advs.201500095.pdf
│   ├── 10.1002advs.201500250.pdf
│   ├── 10.1002advs.201600032.pdf
│   ├── 10.1002advs.201600117.pdf
│   ├── 10.1002advs.201903419.pdf
│   ├── 10.1002advs.202000149.pdf
│   ├── 10.1002advs.202001986.pdf
│   ├── 10.1002advs.202104613.pdf
│   ├── 10.1002advs.202203513.pdf
│   ├── 10.1002advs.202302880.pdf
│   ├── 10.1002aelm.201800915.pdf
│   ├── 10.1002aelm.202300422.pdf
│   ├── 10.1002aenm.201601225.pdf
│   ├── 10.1002aenm.201700390.pdf
│   ├── 10.1002aenm.201700519.pdf
│   ├── 10.1002aenm.201701073.pdf
│   ├── 10.1002aenm.201701201.pdf
│   ├── 10.1002aenm.201701942.pdf
│   ├── 10.1002aenm.201702831.pdf
│   ├── 10.1002aenm.201702941.pdf
│   ├── 10.1002aenm.201703058.pdf
│   ├── 10.1002aenm.201800550.pdf
│   ├── 10.1002aenm.201802050.pdf
│   ├── 10.1002aenm.201901728.pdf
│   ├── 10.1002aenm.201903609.pdf
│   ├── 10.1002aenm.202001203.pdf
│   ├── 10.1002aenm.202001589.pdf
│   ├── 10.1002aenm.202003141.pdf
│   ├── 10.1002aenm.202102135.pdf
│   ├── 10.1002aenm.202200641.pdf
│   ├── 10.1002aenm.202300249.pdf
│   ├── 10.1002aenm.202300980.pdf
│   ├── 10.1002anie.201806354.pdf
│   ├── 10.1002anie.202115585.pdf
│   ├── 10.1002app.45399.pdf
│   ├── 10.1002asia.201100419.pdf
│   ├── 10.1002chem.202002632.pdf
│   └── 10.1002cphc.200901023.pdf
├── pytest.ini
├── requirements.txt
├── scripts
│   └── update_readme_tree.py
└── storage
    └── ontology
        ├── opv_ontology_DeepSeek70b_CBORG.ttl
        ├── opv_ontology_after_extraction.owl
        ├── opv_ontology_after_extraction.ttl
        ├── opv_ontology_current.owl
        ├── opv_ontology_current.ttl
        ├── opv_ontology_phase1.owl
        ├── opv_ontology_phase1.ttl
        ├── opv_ontology_updated.owl
        └── opv_ontology_updated.ttl

13 directories, 107 files
</pre>
<!-- TREE END -->


## Features

Included in this template are a number of helpful things to get you started on the ground running.

### GitHub Actions `.github/workflows/build-app.yml`

Automate linting, pytest, and mkdocs when you push changes to GitHub.

### MkDocs

Create nice documentation with MkDocs and deploy it directly in your repository (Note: Your repository must be set to `public`).

### `.gitignore`

Already configured with a number of common files to ignore.

### `requirements.txt`

List of Python dependencies, such as flake8, pytest, and mkdocs.

### flake8

Lint your Python code for errors with flake8.

### PyTest

Write unit tests with PyTest and they will run when you submit a push to GitHub.

## LBNL Software Disclosure and Distribution

[Here is the official lab policy regarding software disclosure and distribution,](https://commons.lbl.gov/display/rpm2/Software+Disclosure+and+Distribution#SoftwareDisclosureandDistribution--1898802862) and below you will find a summarized version. It is general good practice to keep your projects marked as `private` until you properly disclose your software through the lab.

- **Purpose:**  
  Ensure DOE compliance by reporting all software intended for external distribution to the Intellectual Property Office (IPO).

- **Who Must Comply:**  
  Berkeley Lab software developers and affiliates (employees, faculty, and on-site collaborators).

- **When to Report:**  
  - Before distributing any new or modified software.
  - Exemptions: Already disclosed or minor updates (<25% change without added functionality).

- **Key Requirements:**  
  - **Submission:** Complete a Software Disclosure form prior to external distribution.
  - **Licensing:**  
    - Obtain appropriate license agreements through IPO.
    - Prefer permissive licenses (BSD, MIT) over proprietary or viral open source licenses (e.g., GNU GPL).
  - **Documentation:**  
    - Record third-party licenses, contributor information, and funding sources.
  - **Tracking:**  
    - If distributed via personal repositories or websites, track and report download/licensing metrics annually.

- **IPO Responsibilities:**  
  Review disclosures, secure DOE approvals, manage licensing agreements, and maintain records.

- **Contact:**  
  For questions, reach out to the Licensing Manager at [ipo@lbl.gov](mailto:ipo@lbl.gov).

