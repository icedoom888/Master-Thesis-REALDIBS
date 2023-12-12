# 9. dataset-structure

Date: 2020-03-25

## Status

In dicussion

## Context

The project is meant to grow, so proper dataset structure is crucial.
The overall goal is t have a version-system for the data, tfds records and trained models.
The bare minimum needs a file system, with folders per version, and a way to create tfds
cache files based on a version. then a trained model refers to this data version as well
as the code timestamp to clearly be able to reproduce the results: same data, same code.

milestone 1: filesystem (current status)
there exists folders with versions as /0/ or /7/ indicating the incremental version.
the folders contain the entire copy of each data for simplicity since the data is small.

The data is located on the storage system: //drz-bb8:/topogang/data/ in three stages:
```
├── data
│   ├── primary_hdd             - raw data
│   ├── secondary_files         - processed data
│   └── tertiary_tfds           - tfds records
```

A tree view of the proposal:
```
└── drzasset
    ├── meta_type
    │   └── 0
    │       ├── 3d
    │       ├── blanks
    │       ├── logoiconbutton
    │       ├── movieposterflyer
    │       ├── photo
    │       ├── slidespaper
    │       └── texture
    │   └── 1
    │       ├── 3d
    │       ├── blanks
    │       ├── logoiconbutton
    │       ├── movieposterflyer
    │       ├── photo
    │       ├── slidespaper
    │       └── texture
    └── meta_type_tiny
        └── 0
            ├── 3d
            ├── movieposterflyer
            └── slidespaper
```

The output results are stored under //drz-bb8:/topogang/noicebox/:

drzasset/secondary_models/<probem_domain>/<classfier>/<version>/<inference_tool>/config.json
drzasset/secondary_models/<probem_domain>/<classfier>/<version>/<inference_tool>/model.h5
drzasset/secondary_models/<probem_domain>/<classfier>/<version>/<inference_tool>/classes.json
drzasset/secondary_models/<probem_domain>/<classfier>/<version>/<inference_tool>/confusion.png
drzasset/secondary_models/<probem_domain>/<classfier>/<version>/<inference_tool>/results.txt

A tree view of the proposal:
```
├── drzasset
│   ├── secondary_models
│   │   ├── asset_class
│   │   │   ├── generictag
│   │   │   │   └── 20191022_000000
│   │   │   │   │    └── keras
│   │   │   │   │       ├── config.json
│   │   │   │   │       └── model.h5
│   │   │   │   │       └── classes.json/npz
│   │   │   │   │       └── confusion.png
│   │   │   │   │       └── results.txt
│   │   │   ├── issuetype
│   │   │   │   └── 20190819_020000
│   │   │   │   │    └── keras
│   │   │   │   │       ├── config.json
│   │   │   │   │       └── model.h5
│   │   │   │   │       └── classes.json/npz
│   │   │   │   │       └── confusion.png
│   │   │   │   │       └── results.txt
│   │   │   ├── metatype
│   │   │   │   └── 20190819
│   │   │   │   │   └── keras
│   │   │   │   │   │   ├── config.json
│   │   │   │   │   │   └── model.h5
│   │   │   │   │   └── tfserving
│   │   │   │   │       ├── config.json
│   │   │   │   │       └── model.h5
│   │   │   ├── orientationtype
│   │   │   └── texturetag
│   │   └── asset_keyword
└── noice_paper
```




## Decision

The decision is for a roadmap to get to the final code of an (cloud) storage service
where data and models are pushed and pulled with consistent versionening, automatically.

milestone 1: filesystem (current status)
there exists folders with versions as /0/ or /7/ indicating the incremental version.
the folders contain the entire copy of each data for simplicity since the data is small.
data is semi-automatically copied/written to the specific folders.

milestone 2: configurable json
still file system yet some json config to a) list the file2class association and
b) keep versions of the system.
data is automatically copied/written to the specific folders.

milestone 3: something

milestone 4: cloud storage

milestone 5: complete dataset management system

## Consequences

What becomes easier or more difficult to do and any risks introduced by the
change that will need to be mitigated.
* easier versioning
* difficult to verify files once no longer a file system
* automatic storage saves a heck of time
