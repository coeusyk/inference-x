# Design: add-playground-eval

## Overview

Introduce a small UI or demo surface that consumes the existing API and does not own backend logic.

## Design goals

- keep UI separate from inference logic
- make demos and article screenshots easy to produce
- keep the backend stable while adding usability

## Planned structure

- `playground/web/`
- lightweight client calls to the existing API
- optional compare flow and benchmark trigger path

## Validation

- UI can call the existing endpoint
- comparisons are visible and repeatable
- no backend contract changes are required