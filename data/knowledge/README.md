# Synthetic Operational Knowledge Corpus

This directory contains the sample knowledge base used to develop and evaluate OpsKnowledge AI.

## All content is synthetic

Every document in this directory is **synthetic and fictional**. The runbooks and incident reports were written for development and evaluation purposes only. The services, clusters, nodes, incident IDs, dates, timings, impact figures, and people or team roles are invented. Hostnames use the reserved `example.com` name, and no document contains real credentials, real customer data, or real infrastructure identifiers.

The documents do not describe any real organization, system, or event. Any resemblance to real incidents is coincidental.

## Runbooks and incident reports

The corpus contains two kinds of documents. They are kept in separate directories because they serve different purposes.

| Kind | Directory | Purpose |
| ---- | --------- | ------- |
| Runbook | `runbooks/` | A general, reusable procedure for diagnosing and resolving a class of problem. It is not tied to a particular event. |
| Incident report | `incidents/` | A record of one specific past event: what happened, the impact, the timeline, the root cause, how it was resolved, and what follow-up work was agreed. |

A runbook answers "how do I troubleshoot this kind of problem?". An incident report answers "what happened the last time, and why?".

## Contents

Runbooks:

| File | Topic |
| ---- | ----- |
| `runbooks/kubernetes-crashloopbackoff.md` | Kubernetes `CrashLoopBackOff` troubleshooting |
| `runbooks/kubernetes-imagepullbackoff.md` | Kubernetes `ImagePullBackOff` troubleshooting |
| `runbooks/linux-disk-space-exhaustion.md` | Linux disk space and inode exhaustion, including Kubernetes node disk pressure |
| `runbooks/http-502-behind-load-balancer.md` | HTTP 502 errors behind a load balancer |

Incident reports:

| File | Incident | Summary |
| ---- | -------- | ------- |
| `incidents/inc-2025-001-report-service-imagepullbackoff-node-disk-full.md` | INC-2025-001 | `ImagePullBackOff` caused by a full node disk, not by the registry |
| `incidents/inc-2025-002-storefront-502-idle-timeout-mismatch.md` | INC-2025-002 | Intermittent 502 errors caused by a keep-alive timeout mismatch with healthy backends |
| `incidents/inc-2025-003-search-api-502-oomkilled-crashloop.md` | INC-2025-003 | 502 errors caused by `OOMKilled` pods in `CrashLoopBackOff` |
| `incidents/inc-2025-004-payments-gateway-crashloopbackoff-config-error.md` | INC-2025-004 | `CrashLoopBackOff` caused by an invalid configuration value |

## Document structure

Documents are plain Markdown. They do not use YAML front matter or any other metadata format that needs a parser.

Runbooks use the following second-level sections, in this order:

1. Overview
2. Symptoms
3. Diagnosis
4. Resolution
5. Verification

Commands in runbooks are labeled. Diagnosis and Verification contain **read-only** commands; the few steps in those sections that create a resource are explicitly labeled **Changes state**. In Resolution, every step that modifies state is labeled **Changes state**, and every step that deletes or discards data is labeled **Destructive**. The commands are illustrative examples for a human operator and contain placeholders such as `<namespace>` or `$REGISTRY_USERNAME`, never real values.

The runbooks describe actions for a human operator. OpsKnowledge AI itself is read-only and does not execute them.

Incident reports start with a list of fields (`Incident ID`, `Affected service`, `Severity`, `Date`, `Status`) and a one-line notice that the report is synthetic, followed by these second-level sections:

1. Impact
2. Timeline
3. Root cause
4. Resolution
5. Follow-up actions

## Intended topical overlap

The documents deliberately share topics and vocabulary so that they can be used later to check retrieval behavior. This overlap is intended for future retrieval verification. It is described here for reference and does not mean that any retrieval or evaluation functionality exists yet.

- **Related runbook and incident:** each runbook has at least one incident report on the same topic.
- **Same symptom, different root cause:**
  - INC-2025-002 and INC-2025-003 are both HTTP 502 incidents. One has healthy backends with a keep-alive timeout mismatch; the other has crashing backends.
  - INC-2025-003 and INC-2025-004 are both `CrashLoopBackOff` incidents. One is an out-of-memory kill; the other is an invalid configuration value.
  - INC-2025-001 shows `ImagePullBackOff` whose root cause is a full node disk, not registry access.
- **Documents that span several topics:**
  - INC-2025-001 relates to the `ImagePullBackOff` and disk-space runbooks.
  - INC-2025-003 relates to the HTTP 502 and `CrashLoopBackOff` runbooks.

## Adding documents

When adding documents to this corpus:

- Keep all content synthetic. Do not add proprietary, confidential, employer-specific, or customer-specific information, real operational data, or content copied from internal systems or tickets.
- Do not include credentials, tokens, keys, connection strings, or real hostnames or addresses. Use `example.com`, `example.internal`, and the documentation IP ranges (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`).
- Follow the structure described above and keep file names stable and descriptive.
