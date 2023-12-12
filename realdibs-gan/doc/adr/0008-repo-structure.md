# 8. repo-structure

Date: 2020-01-31

## Status

In dicussion

## Context

The project is meant to grow, so proper folder / module structure is crucial.

## Decision

A tree view of the poposal:
```
.
├── bitbucket-pipelines.yml
├── commit.sh  -------------------------------------   TO REMOVE
├── environment-dev.yml
├── environment.yml
├── README.rst
├── run.sh   --------------------------------------- TO REMOVE (we have bin/..)
│
├── setup.cfg
├── setup.py
│
├── doc
│   ├── adr
│   │   ├── 0001-record-architecture-decisions.md
│   │   ├── 0002-language-and-core-dependencies.md
│   │   ...
│   │   ├── 0007-separeted-regular-and-dev-environment-yml.md
│   │   └── 0008-repo-structure.md
│   ├── api.rst
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   ├── quick_start.rst
│   │   ...
│   ├── _static
│   │   ├── css
│   │   │   └── noice-toolbox.css
│   │   └── js
│   │       └── copybutton.js
│   └── _templates
│       ├── class.rst
│       ├── function.rst
│       └── numpydoc_docstring.py
│
├── bin
│   ├── my_executable_name.py
│   └── README.txt
│
├── examples
│   ├── plot_sine.py
│   └── README.txt
│
├── noice
│   ├── __init__.py
│   ├── _version.py
│   ├── foo.py
│   ├── networks.py
│   │   ...
│   ├── datasets
│   │   ├── base.py
│   │   ├── utils.py
│   │   ├── __init__.py
│   │   └── my_facy_dataset.py
│   │   ...
│   └── utils
│       ├── base.py
│       ├── config.py
│       └── fancy_utils.py
│
└── tests
    ├── __init__.py
    ├── test_network.py
    ...
    ├── test_datasets.py
    └── test_utils.py
```

## Consequences

What becomes easier or more difficult to do and any risks introduced by the change that will need to be mitigated.