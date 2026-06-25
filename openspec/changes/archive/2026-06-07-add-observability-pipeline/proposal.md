# Proposal: add-observability-pipeline

## Summary

Add metrics, logging, and trace-friendly recording so InferenceX can be measured without changing the API contract.

## Why

A serving platform without observability cannot be tuned, debugged, or compared reliably.

## In scope

- request logging
- latency metrics
- token usage capture
- simple storage or exporter path
- observability middleware

## Out of scope

- dashboard UI
- heavy analytics infrastructure
- model training pipelines
- contract changes to the chat API