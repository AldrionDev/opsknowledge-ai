# INC-2025-002: Intermittent HTTP 502 errors on storefront caused by keep-alive timeout mismatch

- **Incident ID:** INC-2025-002
- **Affected service:** storefront
- **Severity:** SEV-2
- **Date:** 2025-05-06
- **Status:** Resolved

This is a synthetic incident report created for development and evaluation. All names, times, and values are fictional.

## Impact

For about 42 hours after a framework upgrade (Day 1 16:30 to Day 3 10:45), on average between 0.4% and 0.9% of requests (with short peaks of up to 1.1%) to `storefront` through the public load balancer returned `HTTP 502 Bad Gateway`. Errors were intermittent and mostly occurred in the first requests after a quiet period, so they were most frequent during low-traffic hours. Checkout page loads and the `/api/cart` endpoint were both affected. Customer support received 43 contacts about "random error pages", and an estimated 600 checkout attempts were retried by customers. The error budget of the service for the month was reduced by 38%.

All `storefront` pods were running and passed their health checks the whole time.

## Timeline

All times are UTC.

- **Day 1, 14:00** `storefront` 3.2.0 is released. It includes an upgrade of the Node.js runtime and the HTTP framework.
- **Day 1, 16:30** The 5xx rate on the load balancer rises from 0.02% to about 0.5%. No alert fires, because the alert threshold is 1% averaged over 5 minutes, and the average stayed below that.
- **Day 2, 09:10** A support ticket from a customer reports intermittent error pages. The on-call engineer starts the investigation.
- **Day 2, 09:40** The pods show no restarts and no `OOMKilled` events, and `kubectl get endpoints storefront` lists all 6 pods. This rules out crashing backends and rollout problems; the CrashLoopBackOff runbook did not apply.
- **Day 2, 10:15** The load balancer access log shows `502` responses with no backend status code and a backend response time of 0 for all failing requests. The storefront application logs contain no matching entries, so the failing requests never reached the application.
- **Day 2, 11:00** Direct requests from a pod in the cluster to the `storefront` Service did not fail in repeated tests, including requests sent after idle gaps on new connections. This indicates that the backend itself is healthy and points to connection reuse between the load balancer and the backend.
- **Day 2, 11:30** The failing requests are correlated with a gap since the previous request on the same connection of about 5 seconds to 60 seconds. The load balancer idle timeout is 60 seconds. A packet capture on one pod shows the pod sending `FIN` on idle connections after about 5 seconds. In the failing cases, the capture shows the load balancer sending a new request on a connection that the pod had already closed and the pod answering with a `RST`. Other requests on idle connections succeeded, for example when the load balancer opened a new connection.
- **Day 2, 13:00** The framework upgrade is identified as the trigger. The previous framework version had configured a keep-alive timeout of 75 seconds explicitly. The new version no longer set it, and the Node.js default of 5 seconds applied.
- **Day 3, 10:00** A change sets `keepAliveTimeout` to 65 seconds and `headersTimeout` to 66 seconds, and a rolling restart is performed.
- **Day 3, 10:45** The 502 rate drops to the baseline. Monitoring continues for 24 hours.
- **Day 4, 11:00** The incident is marked resolved.

## Root cause

The backend HTTP keep-alive (idle) timeout of `storefront` (5 seconds, the Node.js default) was **shorter** than the idle timeout of the load balancer (60 seconds). The load balancer keeps idle connections to the backend open and reuses them for later requests. The backend closed idle connections after 5 seconds, and the load balancer was not always aware of it. When the load balancer sent a request over a connection that the backend had already closed, the backend answered with a TCP reset, and the load balancer returned `502` to the client without the request reaching the application. A mismatch does not fail every request: many requests succeeded because they arrived on fresh or still-open connections, or because the load balancer opened a new connection.

The framework upgrade removed an explicit keep-alive setting (75 seconds) and silently changed the effective timeout. The problem was intermittent because it only occurs when a request arrives within a short window around the moment the backend closes an idle connection, so the failure rate depended on traffic patterns.

Contributing factors:

- The relationship between the load balancer idle timeout and the backend keep-alive timeout was not documented and not tested.
- The alert threshold of 1% did not catch a sustained 0.5% error rate.
- The upgrade release notes were not reviewed for changed defaults.

## Resolution

- Set the backend `keepAliveTimeout` to 65 seconds, above the 60-second load balancer idle timeout, and `headersTimeout` to 66 seconds.
- Performed a rolling restart of all `storefront` pods.
- Confirmed that the load balancer 502 rate returned to baseline and that every access log entry had a backend status code.

The HTTP 502 runbook section on keep-alive timeout mismatch describes this cause and the correct direction of the fix. Lowering the load balancer idle timeout was rejected because the same load balancer serves other services.

## Follow-up actions

- [x] Set the keep-alive and headers timeouts explicitly in the `storefront` server code, with a comment that links them to the load balancer idle timeout.
- [x] Lower the 5xx alert threshold to 0.3% over 30 minutes for `storefront`.
- [ ] Add the load balancer idle timeout and the required backend keep-alive timeout to the service onboarding checklist.
- [ ] Add a post-deploy check that compares the effective backend keep-alive timeout with the load balancer idle timeout.
- [ ] Review dependency upgrade notes for changes to HTTP server defaults as part of the upgrade process.
- [ ] Add a load balancer 502-without-backend-status panel to the `storefront` dashboard.
