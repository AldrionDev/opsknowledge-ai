# INC-2025-001: report-service pods stuck in ImagePullBackOff after node disk filled up

- **Incident ID:** INC-2025-001
- **Affected service:** report-service
- **Severity:** SEV-3
- **Date:** 2025-03-18
- **Status:** Resolved

This is a synthetic incident report created for development and evaluation. All names, times, and values are fictional.

## Impact

During a routine release of `report-service` (version 2.14.0), the new pods scheduled to worker node `node-a3` in `cluster-a` could not start and stayed in `ImagePullBackOff`. The rollout stalled at 2 of 4 updated replicas for about 62 minutes (09:08 to 10:10). Four replicas stayed ready throughout (two on the old version and two on the new version), so end users saw no errors. However, the rollout was incomplete when the daily finance report run started at 09:30, so the run executed on a mixed-version fleet, failed validation on the two old replicas, and was repeated after the rollout completed. The daily finance reports were delivered about 60 minutes late. No data was lost.

The release was blocked and a second, unrelated hotfix for the same service had to wait until the rollout completed.

## Timeline

All times are UTC.

- **09:05** The release pipeline deploys `report-service:2.14.0`. The Deployment has 4 replicas and uses the `RollingUpdate` strategy with `maxSurge: 50%` and `maxUnavailable: 0`, so up to 6 pods can exist during the rollout. The rollout starts.
- **09:08** Two new pods start normally on `node-a1` and `node-a2`. Two new pods scheduled to `node-a3` show `ErrImagePull`, then `ImagePullBackOff`.
- **09:12** The deployment alert "rollout not progressing" fires. The on-call engineer opens the incident.
- **09:15** The on-call engineer suspects a registry or credential problem, since `ImagePullBackOff` is the typical symptom. The image tag `2.14.0` is confirmed to exist in `registry.example.com`, and pods on the other two nodes pulled it successfully, which rules out a wrong tag and expired pull credentials.
- **09:24** `kubectl describe pod` shows the event `Failed to pull image ... write /var/lib/containerd/...: no space left on device`. This points at the node, not at the registry.
- **09:31** On `node-a3`, `df -h` shows `/var` at 100% usage. `kubectl describe node node-a3` reports `DiskPressure=True`, which appeared at 09:10 because the failed pulls left partially downloaded layers behind. Before the release the node had about 18% free space, which is above the default kubelet hard eviction thresholds, so it was still schedulable, but the new image needed more space than was free while its layers were downloaded and extracted. The kubelet image garbage collection could not free enough space (`FreeDiskSpaceFailed` events since 09:10).
- **09:40** `du` shows that `/var/log/pods` and `/var/log/app-archive` use 71% of the filesystem. An archived log directory from a batch job named `report-export` had grown to 38 GB because log rotation for that directory was never configured.
- **09:48** The node is cordoned to prevent further scheduling.
- **09:55** The on-call engineer compresses and moves the archived logs to object storage after confirming with the service owner that they are not needed locally, then removes unused container images with `crictl rmi --prune`. Disk usage drops to 54%.
- **10:02** `DiskPressure` clears. The node is uncordoned.
- **10:05** The two stuck pods pull the image, start, and become ready. The rollout completes at **10:10**.
- **10:30** The repeated finance report run finishes.
- **10:40** The incident is marked resolved after monitoring stays normal for 30 minutes following the rollout completion at 10:10.

## Root cause

The root cause was an unbounded log archive on the worker node. A batch job (`report-export`) wrote logs to `/var/log/app-archive` on the node, and no log rotation or retention rule existed for that path. Over several months the archive filled the shared `/var` filesystem. The kubelet image garbage collection could not free enough space, and because the filesystem was full, the container runtime could not write the layers of the new `report-service` image.

The symptom (`ImagePullBackOff`) suggested a registry or credential issue, which delayed the diagnosis by about 10 minutes. The decisive clue was that only one node was affected and that the pull error text contained `no space left on device`.

Contributing factors:

- `/var` was a single filesystem shared by container images, pod logs, and application archives, with no separate volume for logs.
- The disk usage alert threshold was set to 95%, and the alert had no predicted-time-to-full rule, so it did not fire before the release. It fired only after the failed pulls had pushed usage over the threshold, and it was routed to a low-priority channel.
- Node-level `DiskPressure` was not part of the deployment health dashboard.

## Resolution

- Cordoned `node-a3` to prevent further scheduling.
- Compressed and moved the archived logs off the node after owner confirmation, and deleted the local copies.
- Removed unused container images on the node with `crictl rmi --prune`.
- Waited for `DiskPressure` to clear, uncordoned the node, and let the Kubernetes controller retry the pulls. Restarting the rollout was not necessary.

The Kubernetes ImagePullBackOff runbook and the Linux disk-space exhaustion runbook describe the diagnosis and remediation steps used in this incident.

## Follow-up actions

- [x] Add a logrotate rule with a 7-day retention for `/var/log/app-archive` on all nodes of `cluster-a`.
- [x] Lower the filesystem usage alert threshold to 80% and add a predicted-time-to-full alert, routed to the on-call channel.
- [ ] Move pod logs and application archives to a separate volume from the container runtime storage. Owner: platform team.
- [ ] Add node `DiskPressure` conditions and image pull failures to the deployment health dashboard.
- [ ] Update the release checklist to compare the failing pod's node with the nodes of healthy pods before investigating registry credentials.
- [ ] Change the `report-export` job to write archives directly to object storage.
