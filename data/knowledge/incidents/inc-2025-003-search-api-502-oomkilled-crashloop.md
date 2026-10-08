# INC-2025-003: search-api 502 errors caused by OOMKilled pods in CrashLoopBackOff

- **Incident ID:** INC-2025-003
- **Affected service:** search-api
- **Severity:** SEV-2
- **Date:** 2025-07-22
- **Status:** Resolved

This is a synthetic incident report created for development and evaluation. All names, times, and values are fictional.

## Impact

For 41 minutes after a release (13:06 to 13:47), `search-api` returned `HTTP 502 Bad Gateway` for between 12% and 35% of requests through the public load balancer. Product search on the web site and the mobile app showed "something went wrong" pages, and autocomplete was unavailable. An estimated 9% decrease in orders was observed during the window. Other services were not affected.

## Timeline

All times are UTC.

- **13:00** `search-api` 5.8.0 is released with a new relevance ranking feature. The rolling update starts.
- **13:06** The load balancer 5xx rate rises from 0.03% to 12%. The "5xx rate high" alert fires at **13:09** and pages the on-call engineer.
- **13:12** The on-call engineer sees 502 responses with no backend status in the load balancer access log and starts with the HTTP 502 runbook. Target health shows 3 of 8 targets unhealthy and the number of ready endpoints changing every few seconds.
- **13:18** `kubectl get pods` shows new `search-api` pods in `CrashLoopBackOff`, with restart counts of 4 to 6, while old-version pods are still running. Pods flap between `Running` and `OOMKilled`.
- **13:22** `kubectl describe pod` shows `Last State: Terminated`, `Reason: OOMKilled`, `Exit Code: 137`. The memory limit of the container is 1Gi. Application logs from `kubectl logs --previous` end abruptly during index loading, with no application error.
- **13:26** Monitoring shows memory usage of the new pods growing to the 1Gi limit within 70 seconds of startup, during which the pod is added to the endpoints because the readiness probe only checks that the HTTP port is open, not that the index is loaded. Requests routed to starting pods fail when the container is killed, and the load balancer returns 502.
- **13:30** The release is identified as the trigger. The new ranking feature loads a larger in-memory model, which increases the startup memory requirement from about 700Mi to about 1.6Gi.
- **13:34** The decision is made to roll back, since a limit change would need capacity checks. `kubectl rollout undo` is run.
- **13:41** The rollback completes. Restart counts stop increasing. The 502 rate starts to fall.
- **13:47** The 5xx rate is back to baseline. Impact ends.
- **Next day** The corrected release (5.8.1) is deployed with a memory limit of 2Gi and request of 1.75Gi, after checking node capacity. A readiness probe that waits for the index to load is added.

## Root cause

The new release increased the memory needed at startup beyond the container limit of 1Gi. The kernel killed the container with `OOMKilled` (exit code 137), and the kubelet restarted it, which led to `CrashLoopBackOff` with growing restart delays.

The user-visible 502 errors were caused by two weaknesses working together:

1. The memory limit of the container was not reviewed or load-tested for the new release.
2. The readiness probe only checked that the TCP port was open, so pods were registered as ready targets on the load balancer while they were still loading the index. The kill, or a reset of connections to a pod that was shutting down, produced the 502 responses.

The cause of the 502 spike was therefore a crashing backend, not a load balancer configuration problem: the load balancer idle timeout and the keep-alive timeout of the application were unrelated to this incident, in contrast to INC-2025-002, where the pods were healthy.

Contributing factors:

- The pre-production environment ran with a larger memory limit than production, so the problem was not reproduced there.
- The rollout used `maxUnavailable: 25%` and did not wait for sustained readiness before continuing, so the number of failing pods grew before the rollout stopped.
- No alert existed for `OOMKilled` events.

## Resolution

- Rolled back the Deployment with `kubectl rollout undo`, which restored the previous release and its memory profile and ended the customer impact.
- Released `search-api` 5.8.1 with a memory limit of 2Gi and request of 1.75Gi, after verifying that the nodes had capacity for the new requests.
- Changed the readiness probe so that a pod only becomes ready after the index is loaded, and added a `startupProbe` so that the liveness probe does not run during the slow startup.

The Kubernetes CrashLoopBackOff runbook (OOMKilled branch) and the HTTP 502 runbook (crashing or not ready backend branch) describe the diagnosis and remediation used.

## Follow-up actions

- [x] Roll back to the previous release and confirm recovery.
- [x] Release 5.8.1 with the corrected memory limit, readiness probe, and startup probe.
- [ ] Align pre-production resource limits with production.
- [ ] Add a load test to the release pipeline that measures peak memory at startup and under normal traffic, and fails the release if usage exceeds 80% of the limit.
- [ ] Add an alert on `OOMKilled` container terminations and on restart rate per workload.
- [ ] Set `maxUnavailable: 0` and `minReadySeconds` for the `search-api` Deployment so a failing release stops before it reduces capacity.
- [ ] Document the memory profile of the ranking model in the service runbook.
