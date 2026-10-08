# INC-2025-004: payments-gateway in CrashLoopBackOff after invalid ConfigMap value

- **Incident ID:** INC-2025-004
- **Affected service:** payments-gateway
- **Severity:** SEV-2
- **Date:** 2025-09-09
- **Status:** Resolved

This is a synthetic incident report created for development and evaluation. All names, times, and values are fictional.

## Impact

After a configuration change, all replicas of `payments-gateway` restarted and then failed to start. For about 25 minutes (10:03 to 10:28) the service had no ready replicas, and payment authorization requests from the checkout service failed with connection errors. Checkout reported payment failures for the whole period. Customers could still browse and fill their carts. No payment was charged twice or lost, since the checkout service rejected the order when authorization failed, and the failed orders could be retried by customers afterwards. About 1,900 checkout attempts failed.

## Timeline

All times are UTC.

- **10:02** An engineer merges a change that updates the `payments-gateway-config` ConfigMap to set `UPSTREAM_TIMEOUT` and `RETRY_BACKOFF_MS`. The change is applied by the deployment pipeline, followed by `kubectl rollout restart` so the pods pick up the new environment values.
- **10:03** The Deployment uses the `Recreate` strategy, so the restart first terminates all old pods and only then creates new ones. The service has no ready replicas from this point.
- **10:04** The new pods exit within a second of starting and enter `CrashLoopBackOff`.
- **10:07** The alert "payments-gateway unavailable" fires because the Deployment has had 0 ready replicas for several minutes.
- **10:11** The on-call engineer sees pods in `CrashLoopBackOff` with `Exit Code: 1` and a restart count of 5. The last state is `Error`, not `OOMKilled`, and the container ran for under one second each time, which excludes an out-of-memory kill and a liveness probe failure.
- **10:14** `kubectl logs --previous` shows `FATAL: invalid configuration: RETRY_BACKOFF_MS must be an integer, got "250ms"`. The application validates its configuration at startup and exits with code 1 on invalid input.
- **10:18** The diff of the ConfigMap shows that `RETRY_BACKOFF_MS` was changed from `250` to `250ms`. The application expects a number of milliseconds with no unit.
- **10:21** The team decides to restore service first by rolling back instead of editing the live ConfigMap. The ConfigMap change is reverted in version control and applied.
- **10:24** `kubectl rollout restart` is run again. The new pods start, pass their readiness probe, and register with the service.
- **10:28** All replicas are ready and payment authorization success returns to normal.
- **10:45** The incident is marked resolved after monitoring is normal for 15 minutes.

## Root cause

A configuration value was changed to a format that the application could not parse (`250ms` instead of `250`). The application validates configuration at startup and exits with code 1 for invalid values. Because environment variables from a ConfigMap are only read when a container starts, the invalid value had no effect until the rollout restart. Every new pod then crashed immediately and entered `CrashLoopBackOff`.

This is a configuration error and not a resource problem: the exit code was 1 (application error), the container was not killed by the kernel, and the memory usage was low. This distinguishes the incident from INC-2025-003, where an out-of-memory kill caused the restart loop.

Contributing factors:

- ConfigMap changes were not validated against the application's configuration schema before being applied.
- The Deployment used the `Recreate` strategy, which terminates all old pods before new pods are created. A bad configuration therefore took the service down completely, instead of the rollout stopping at the first failing new pod while old pods kept serving.
- The configuration change did not pass through a staging environment.
- The unit convention (milliseconds without suffix) was not documented next to the setting.

## Resolution

- Reverted the ConfigMap change in version control and applied the previous value.
- Ran `kubectl rollout restart deployment/payments-gateway` to pick up the corrected environment values.
- Verified that all replicas were ready, that `kubectl logs` showed a normal startup, and that the authorization success rate recovered.

The Kubernetes CrashLoopBackOff runbook (application configuration error branch) describes the diagnosis and the remediation used. Using `kubectl rollout undo` alone would not have helped, because the ConfigMap is not part of the Deployment revision history.

## Follow-up actions

- [x] Revert the invalid ConfigMap change.
- [ ] Add a CI validation step that checks ConfigMap values against the application's configuration schema before merge.
- [ ] Switch the `payments-gateway` Deployment to the `RollingUpdate` strategy with `maxUnavailable: 0` and `maxSurge: 1`, so that a failing new pod blocks the rollout while the old pods keep serving.
- [ ] Apply configuration changes to staging first and require a successful rollout there.
- [ ] Document the units and allowed values of each configuration setting in the service README.
- [ ] Add a startup log line that reports the names of the settings that failed validation, and an alert for pods with exit code 1 at startup.
- [ ] Consider rolling out the ConfigMap under a versioned name so that a rollback restores the previous configuration automatically.
