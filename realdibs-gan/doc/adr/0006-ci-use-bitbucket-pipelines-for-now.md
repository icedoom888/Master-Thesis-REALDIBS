# 6. CI-use-bitbucket-pipelines-for-now

Date: 2020-01-29

## Status

Accepted

## Context

The CI infrastructure using Buildbot and GPUs is not available yet.

## Decision

Use **bitbucket-pipeline** for the time being.

## Consequences

We will add `.bitbucket-pipeline.yml` to the repo and activate the service from bitbucket.
