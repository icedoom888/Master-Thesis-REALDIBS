# 5. remove-absl

Date: 2020-01-29

## Status

Accepted

## Context

For historical reasons, the project has a dependency on [absl-py](https://pypi.org/project/absl-py/) which is a google internal tool.

From their documentation:

> The Abseil containers are designed to be more efficient in the general
> case; in some cases, however, the STL containers may be more efficient.
> Unlike some other abstractions that Abseil provides, these containers
> should not be considered drop-in replacements for their STL counterparts,
> as there are API and/or contract differences between the two sets of
> containers. For example, the Abseil containers often do not guarantee
> pointer stability after insertions or deletions.

From their code:

```py
# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
```

## Decision

We will be better off by making our own code, or finding a better library to do the same job.

## Consequences

For the time being we'll implement a `Bunch` class (and derivates) to achieve similar functionalities.
This `Bunch` class would go inside `$GIT_ROOT/$PACKAGE_NAME/utils/`
