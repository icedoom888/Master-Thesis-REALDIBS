# 7. separeted-regular-and-dev-environment-yml

Date: 2020-01-29

## Status

Accepted

## Context

Dev environments need more dependencies than whatever is necessary to
strictly run the code. (i.e: in ourder to build the documentation we would
need `sphinx`)

## Decision

Keep necessary, optional, and development dependencies in separate
`environment-xxx.yml` and build the environment in the CI using whatever is
necessary.

## Consequences

Create `environment.yml` and `environment-dev.yml` files.
