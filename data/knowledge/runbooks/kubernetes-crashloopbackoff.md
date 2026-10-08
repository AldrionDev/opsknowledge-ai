# Kubernetes CrashLoopBackOff Troubleshooting

## Overview

`CrashLoopBackOff` is a pod status reported when a container starts, exits, and is restarted repeatedly by the kubelet. The kubelet delays each restart with an exponential back-off (10s, 20s, 40s, and so on). The delay is capped at 300 seconds (5 minutes) by default. In recent Kubernetes versions the maximum can be lowered per node through the kubelet configuration (`crashLoopBackOff.maxContainerRestartPeriod`, between 1s and 300s, behind the `KubeletCrashLoopBackOffMax` feature gate), so the cap on a given cluster may be shorter than 300 seconds. The back-off timer resets after a container runs successfully for 10 minutes.

`CrashLoopBackOff` is a symptom, not a root cause. The container process is the thing that is failing. Common causes are:

- an application error at startup, such as an invalid configuration value or a missing environment variable
- the container being killed for exceeding its memory limit (`OOMKilled`)
- a failing liveness probe that causes the kubelet to restart a container that is actually running
- a missing dependency, such as a ConfigMap, Secret, or database that the process requires at startup
- a wrong container command or entrypoint, or a file permission problem

This runbook applies to workloads managed by a Deployment, StatefulSet, DaemonSet, or Job. It assumes the pod is scheduled and the image was pulled. If the status is `ImagePullBackOff` or `ErrImagePull`, use the Kubernetes ImagePullBackOff runbook instead.

Command conventions used in this runbook:

- **Read-only** commands only inspect state and are safe to run at any time.
- **Changes state** commands modify configuration, workloads, or cluster resources.
- **Destructive** commands delete or discard data and need an explicit check of the target first.

## Symptoms

- `kubectl get pods` shows `CrashLoopBackOff` in the `STATUS` column and a growing `RESTARTS` count.
- The pod alternates between `Running`, `Error`, `Completed`, or `OOMKilled` and `CrashLoopBackOff`.
- A Deployment rollout stalls: `kubectl rollout status` reports that new replicas are not available.
- Alerts fire for high container restart rate or for unavailable replicas.
- Upstream callers see connection errors or HTTP 502/503 responses while replicas restart. See the HTTP 502 runbook if the service is behind a load balancer.

## Diagnosis

All commands in this section are **read-only**. Replace `<namespace>` and `<pod>` with the values for the affected workload.

1. Confirm the status and restart count.

   ```bash
   kubectl get pods -n <namespace> -o wide
   ```

2. Inspect the pod events and the last container state. Look at `Last State`, `Reason`, `Exit Code`, and the `Events` section at the bottom of the output.

   ```bash
   kubectl describe pod <pod> -n <namespace>
   ```

3. Read the logs of the previous (crashed) container instance. The current instance may have no output yet because it was just restarted.

   ```bash
   kubectl logs <pod> -n <namespace> --previous
   ```

   For a multi-container pod, add `-c <container>`.

4. Extract the termination reason and exit code directly.

   ```bash
   kubectl get pod <pod> -n <namespace> \
     -o jsonpath='{.status.containerStatuses[*].lastState.terminated}'
   ```

5. Interpret the exit code.

   | Exit code | Meaning | Typical cause |
   | --------- | ------- | ------------- |
   | 0 | Process exited normally | Container runs a short command and is not a long-running process, or a Job template is used where a Deployment was intended |
   | 1 | Generic application error | Invalid configuration, unhandled exception at startup, failed dependency check |
   | 126 / 127 | Command not executable / not found | Wrong `command` or `args`, missing binary, wrong image |
   | 137 | Killed by SIGKILL | Usually `OOMKilled` when the reason is `OOMKilled`; can also be a forced kill after a failed liveness probe |
   | 139 | Segmentation fault | Native crash in the application or a library |
   | 143 | Terminated by SIGTERM | Graceful shutdown, for example during a rollout or node drain |

6. If the reason is `OOMKilled`, compare the container memory limit with actual usage.

   ```bash
   kubectl get pod <pod> -n <namespace> \
     -o jsonpath='{.spec.containers[*].resources}'
   kubectl top pod <pod> -n <namespace> --containers
   ```

   `kubectl top` requires the metrics server and only shows current usage, so a container that is killed quickly may not appear. Check the memory graphs of the monitoring system for the peak before the kill.

7. Check the namespace events for scheduling, probe, and volume problems.

   ```bash
   kubectl get events -n <namespace> --sort-by=.lastTimestamp
   ```

   Probe failures appear as `Unhealthy` events, for example `Liveness probe failed`.

8. Verify that the configuration the pod depends on exists and contains the expected keys. Do not print Secret values to a shared terminal or ticket; check key names only.

   ```bash
   kubectl get configmap <name> -n <namespace> -o yaml
   kubectl describe secret <name> -n <namespace>
   ```

9. Check whether the failure started after a change. Compare the current revision with the previous one.

   ```bash
   kubectl rollout history deployment/<deployment> -n <namespace>
   ```

## Resolution

Choose the branch that matches the cause found during diagnosis.

### Application configuration error (exit code 1)

1. Fix the invalid value in the ConfigMap manifest in version control, then apply it. **Changes state:**

   ```bash
   kubectl apply -f configmap.yaml -n <namespace>
   ```

2. Environment variables sourced from a ConfigMap are read only when the container starts, so restart the workload to pick up the change. **Changes state:** this replaces all pods of the Deployment through a rolling update.

   ```bash
   kubectl rollout restart deployment/<deployment> -n <namespace>
   ```

3. If the configuration change cannot be fixed quickly, restore the last known good ConfigMap from version control and restart the workload as above. `kubectl rollout undo` only helps when the change was made to the pod template (for example the image, environment variables defined inline, or a versioned ConfigMap name). A ConfigMap edited in place is not part of the Deployment revision history, so `rollout undo` does not restore it. **Changes state.**

### OOMKilled (exit code 137, reason `OOMKilled`)

1. Decide whether the application has a memory leak or the limit is simply too low for normal load. A leak shows steadily growing memory until the kill; an undersized limit shows a flat profile that hits the limit under normal traffic.
2. For an undersized limit, increase `resources.limits.memory` (and `resources.requests.memory`) in the manifest and apply it. **Changes state:** the change triggers a rolling update.

   ```yaml
   resources:
     requests:
       memory: 512Mi
     limits:
       memory: 1Gi
   ```

3. Confirm that the nodes have enough allocatable memory for the new requests, otherwise pods will stay `Pending`.
4. For a suspected leak, raising the limit only delays the restart. Roll back to the last good release and open a defect for the application team. **Changes state:**

   ```bash
   kubectl rollout undo deployment/<deployment> -n <namespace>
   ```

### Failing liveness probe

1. Run the probe command or request manually against a running container and compare the result with the probe definition in `kubectl describe pod`.
2. If the application needs a long startup, add a `startupProbe` or increase `initialDelaySeconds` and `failureThreshold` so the liveness probe does not run during startup. **Changes state:**

   ```yaml
   startupProbe:
     httpGet:
       path: /healthz
       port: 8080
     periodSeconds: 5
     failureThreshold: 30
   ```

3. Liveness probes should check only the health of the process itself, not external dependencies. A liveness probe that fails when the database is down restarts every replica at once and makes the outage worse.

### Missing dependency or wrong command

1. If a referenced ConfigMap, Secret, or volume is missing, create it from the source of truth in version control. **Changes state.**
2. If the `command` or `args` are wrong (exit codes 126 or 127), correct the pod template and apply it, or roll back with `kubectl rollout undo`. **Changes state.**

### Deleting a stuck pod

Deleting a pod does not fix a crash loop, because the controller recreates it with the same specification. Delete a pod only to clear a pod that is stuck for node-level reasons. **Destructive:** confirm the exact pod name and that the pod is managed by a controller, otherwise it will not be recreated.

```bash
kubectl delete pod <pod> -n <namespace>
```

## Verification

All commands in this section are **read-only**.

1. The pod reaches `Running` with all containers ready, and the restart count stops increasing.

   ```bash
   kubectl get pods -n <namespace> --watch
   ```

2. The rollout completes.

   ```bash
   kubectl rollout status deployment/<deployment> -n <namespace>
   ```

3. Observe the pod for at least two probe periods plus the longest expected startup time. A container that fails after 10 minutes of uptime restarts with a fresh back-off, so a short observation window can miss a slow crash.
4. The new container logs show a normal startup sequence and no repeated errors.

   ```bash
   kubectl logs <pod> -n <namespace> --tail=100
   ```

5. Service-level indicators recover: error rate, latency, and available replica count return to their normal range.
