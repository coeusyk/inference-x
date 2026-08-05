# Platform Specification Delta — add-ci-and-static-analysis-gate

## ADDED Requirements

### Requirement: Changes are verified automatically before merge
The project SHALL verify every proposed change against the unit suite, the
configured lint rules, and the configured type baseline automatically, without
relying on a contributor or reviewer to run those checks by hand.

#### Scenario: A proposed change breaks an existing test
- WHEN a change is proposed that causes any test in the unit suite to fail
- THEN the automated verification reports failure
- AND the change cannot be merged into a long-lived branch until it passes

#### Scenario: A proposed change violates lint or type rules
- WHEN a change is proposed that violates a configured lint rule, or introduces a
  type error in a module outside the recorded type baseline
- THEN the automated verification reports failure
- AND the change cannot be merged into a long-lived branch until it passes

#### Scenario: Verification requires no accelerator
- WHEN automated verification runs
- THEN it completes on a standard hosted runner with no GPU present
- AND no test is skipped solely because verification ran without a GPU

#### Scenario: The type baseline is explicit and bounded
- WHEN a module is exempted from type checking
- THEN that module is named individually in the recorded baseline
- AND no repository-wide or wildcard exemption is configured, so that a module
  added later is checked by default rather than silently exempted

#### Scenario: Change is applied
- WHEN this change is archived
- THEN automated verification runs on pull requests and on pushes to long-lived
  branches
- AND the required checks are enforced at merge time, not merely reported
- AND the type baseline and the policy governing it are recorded in
  `docs/DECISIONS.md`
