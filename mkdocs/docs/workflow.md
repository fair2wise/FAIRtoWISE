flowchart LR
    %% Continuous paper inflow
    Papers[(Scientific Papers)]:::literature --"New Literature"--> Input
    
    %% Main script with Prefect
    Main["main.py(Prefect Workflow)"]:::orchestrator
    
    subgraph Input["Data Ingestion"]
        direction TB
        A["PDF/DOCX/HTMLDocuments"]:::document --> |"Extract Content"| B["Text Extraction(PyMuPDF/Tika)"]:::extraction
    end
    
    subgraph Extract["Knowledge Extraction"]
        direction TB
        C["LLM Service(Ollama/OpenAI/Claude)"]:::llm
        D["Term Extraction(Prompt Engineered)"]:::extraction
        E["Relation Detection(Pattern-Based Prompts)"]:::extraction
        PT["Prompt Templates(YAML/Jinja2)"]:::config
        
        PT -."Configure Prompts".-> C
        C --"Identify Terms"--> D
        C --"Detect Relations"--> E
    end
    
    subgraph Structure["Knowledge Organization"]
        direction TB
        F["SKOS Thesaurus(RDF/Turtle)"]:::knowledge
        G["OWL Ontology(LinkML/ROBOT)"]:::knowledge
        H["Knowledge Graph(Neo4j/RDF)"]:::knowledge
        
        F --"Convert to Ontology"--> G
        G --"Populate Graph"--> H
    end
    
    subgraph Apply["Knowledge Utilization"]
        direction TB
        I1["KG Extraction(SPARQL/Cypher)"]:::extraction
        I2["Vector Indexing(FAISS/HNSW)"]:::knowledge
        I3["Semantic Retrieval"]:::knowledge
        J["LLM API Integration(Ollama/CBORG/OpenAI/Anthropic)"]:::llm
        K["Validation Framework(Human-in-the-Loop)"]:::human
        
        I1 --"Extract Subgraphs"--> I2
        I2 --"Find Relevant Knowledge"--> I3
        I3 --"Context Assembly"--> J
        J --"Generate Answers"--> K
        K -."Feedback Loop".-> I3
    end
    
    %% Orchestration connections
    Main --"Schedule Tasks"--> Input
    Main --"Configure Model"--> Extract
    Main --"Build Schema"--> Structure
    Main --"Manage Queries"--> Apply
    K -."Update Pipeline".-> Main
    
    %% Data flow connections
    Input --"Processed Text"--> Extract
    Extract --"Terms & Relations"--> Structure
    Structure --"Knowledge Base"--> Apply
    
    %% Continuous data updates
    Apply -."Enrich Knowledge".-> Papers
    
    %% Color scheme - more fun and meaningful
    classDef orchestrator fill:#FF6B6B,stroke:#333,stroke-width:3px
    classDef document fill:#4ECDC4,stroke:#333,stroke-width:2px
    classDef extraction fill:#FFE66D,stroke:#333,stroke-width:2px
    classDef knowledge fill:#1A535C,stroke:#FFF,stroke-width:2px,color:#FFF
    classDef llm fill:#7B2CBF,stroke:#FFF,stroke-width:2px,color:#FFF
    classDef config fill:#F7B267,stroke:#333,stroke-width:2px
    classDef human fill:#06D6A0,stroke:#333,stroke-width:2px
    classDef literature fill:#FFA69E,stroke:#333,stroke-width:2px