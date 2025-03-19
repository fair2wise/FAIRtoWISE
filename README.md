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

