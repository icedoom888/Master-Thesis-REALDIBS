# 4. add-ci-for-documentation

Date: 2020-01-24

## Status

Accepted

## Context

Most of the time, down the line projects become inconsistent in documentation.

## Decision

Add tests in the CI in order to ensure that the code is properly documented and follows some general guidelines.
We'll use `pydoc` ..

possible dowbacks: it might slow down the development process in which case we will review the policy.

## Consequences

add pytests.
add CI entry for that.

