# Linux Disk Space Exhaustion

## Overview

A Linux filesystem can run out of space in two different ways: it can run out of **blocks** (data space) or out of **inodes** (the number of files it can hold). Both produce the error `No space left on device` (`ENOSPC`), but they need different diagnosis and different fixes.

Typical consequences on a server or Kubernetes worker node:

- applications fail to write logs, temporary files, or data, and may crash or hang
- databases and message brokers stop accepting writes
- package installs and deployments fail
- on a Kubernetes node, the kubelet reports the `DiskPressure` condition, evicts pods, stops scheduling new pods to the node, and container images can no longer be pulled

Common causes are unrotated or runaway log files, core dumps, a temporary-file directory that is never cleaned, accumulated container images and layers, journald growth, and large files that were deleted while a process still held them open. Disk exhaustion is usually a capacity or housekeeping problem, but a sudden increase often indicates an application defect such as an error loop that writes huge logs.

This runbook covers a single Linux host and the Kubernetes node case. Pod scheduling and image-pull symptoms that result from node disk pressure are described in the Kubernetes ImagePullBackOff runbook.

Command conventions used in this runbook:

- **Read-only** commands only inspect state and are safe to run at any time.
- **Changes state** commands modify files, services, or cluster resources.
- **Destructive** commands delete or discard data and need an explicit check of the target first.

## Symptoms

- Errors such as `No space left on device`, `write failed, filesystem is full`, or `cannot create temp file`.
- Monitoring alerts for filesystem usage above the threshold (for example 85% or 90%) or for a high filesystem fill rate.
- `df -h` shows 100% usage on a filesystem such as `/`, `/var`, or `/var/log`.
- `df -h` shows free space, but file creation still fails. This indicates inode exhaustion (`df -i` shows 100% `IUse%`) or a quota.
- Services crash or restart, and databases switch to read-only mode.
- Kubernetes: node condition `DiskPressure=True`, pods with status `Evicted`, events like `The node was low on resource: ephemeral-storage`, and image pull failures with `no space left on device`.

## Diagnosis

All commands in this section are **read-only**. On a production host, run `du` with `-x` to stay on one filesystem and prefer `nice`/`ionice` because scanning a large filesystem generates I/O load.

1. Identify which filesystem is full, in blocks and in inodes.

   ```bash
   df -hT
   df -i
   ```

2. Find the largest directories on the affected filesystem, descending one level at a time.

   ```bash
   sudo du -xh --max-depth=1 /var | sort -rh | head -n 15
   ```

   Repeat for the largest subdirectory, for example `/var/log` or `/var/lib`.

3. Find the largest individual files.

   ```bash
   sudo find /var -xdev -type f -size +500M -exec ls -lh {} + 2>/dev/null
   ```

4. If blocks are full but `du` accounts for far less than `df` reports, a deleted file is probably still held open by a running process. List open files with a link count of zero.

   ```bash
   sudo lsof +L1
   ```

   The `SIZE/OFF` column and the `COMMAND`/`PID` columns show which process holds the space.

5. If inodes are exhausted, find the directories with the most files.

   ```bash
   sudo find /var -xdev -type f | cut -d/ -f1-4 | sort | uniq -c | sort -rn | head -n 15
   ```

   Typical sources are session files, mail queues, and cache directories with millions of small files.

6. Check the usual space consumers.

   ```bash
   sudo journalctl --disk-usage
   ls -lh /var/log | sort -k5 -h | tail -n 10
   sudo coredumpctl list --no-pager | tail -n 10
   ```

7. Check whether log rotation works. A dry run prints what would happen without rotating anything.

   ```bash
   sudo logrotate -d /etc/logrotate.conf
   ```

   Look for errors such as a missing log directory, wrong permissions, or a rotation that is configured but never triggered.

8. On a Kubernetes node, check the node condition and the container runtime storage.

   ```bash
   kubectl describe node <node>
   df -h /var/lib/containerd /var/lib/kubelet
   sudo crictl images
   sudo crictl ps -a | head -n 20
   ```

   Pod logs under `/var/log/pods` and container writable layers also count towards node disk usage.

9. Establish what changed. Check the growth rate in monitoring and the time of the last deployment. A steep recent increase usually points to a new error loop or a changed log level, while a slow linear increase points to missing rotation or retention.

## Resolution

Work from the safest to the most destructive action. Free enough space to stabilize the host first, then fix the underlying cause so the problem does not return.

### Stabilize: free space safely

1. Compress or move old logs instead of deleting them when the data may be needed for an investigation. **Changes state:**

   ```bash
   sudo gzip /var/log/<application>/<old-log-file>
   ```

2. Reduce the journal to a fixed size. **Destructive:** this deletes the oldest archived journal files until the total falls below the limit. The deleted log history is not recoverable. Active journal files are not affected, so check `journalctl --disk-usage` first and make sure the logs are not needed for an ongoing investigation.

   ```bash
   sudo journalctl --vacuum-size=500M
   ```

3. Remove files that are confirmed to be disposable, such as old core dumps or temporary files. **Destructive:** deleted files cannot be recovered. Confirm the exact path with `ls -l` first and never use a wildcard on a path you have not listed.

   ```bash
   ls -l /var/crash
   sudo rm /var/crash/<specific-core-file>
   ```

4. If a very large log file is still being written by a running process, do not delete it, because the process keeps the space allocated (see the deleted-but-open case). Truncate it in place instead. **Destructive:** this discards the file content.

   ```bash
   sudo truncate -s 0 /var/log/<application>/<large-log-file>
   ```

### Deleted-but-open files

1. Identify the process with `lsof +L1`.
2. Restart or reload the owning service so that it releases the file. **Changes state:** this causes a service interruption, so coordinate with the service owner and prefer a rolling restart behind a load balancer.

   ```bash
   sudo systemctl restart <service>
   ```

### Inode exhaustion

1. Delete the files in the offending directory in a controlled way after confirming they are disposable. **Destructive:** use a narrow filter, for example files older than a retention period.

   ```bash
   sudo find /var/spool/<directory> -xdev -type f -mtime +7 -print
   ```

   Review the printed list, and only then replace `-print` with `-delete`.
2. Fix the producer so that it cleans up after itself or uses a bounded cache. **Changes state.**

### Kubernetes node with DiskPressure

1. Cordon the node so no new pods are scheduled to it while it is repaired. **Changes state:**

   ```bash
   kubectl cordon <node>
   ```

2. Remove unused container images. **Destructive:** this deletes cached images that no container is using. They are pulled again when needed, which adds registry load.

   ```bash
   sudo crictl rmi --prune
   ```

3. Tune the kubelet image garbage collection thresholds (`imageGCHighThresholdPercent`, `imageGCLowThresholdPercent`) and eviction thresholds if the defaults do not leave enough headroom. **Changes state:** this requires a kubelet configuration change and restart on the node.
4. After the filesystem has enough free space and `DiskPressure` clears, uncordon the node. **Changes state:**

   ```bash
   kubectl uncordon <node>
   ```

### Fix the root cause

All items in this branch are **Changes state** (configuration, infrastructure, or code changes) and should go through the normal change process.

- Add or correct log rotation (`logrotate` or journald limits) with a size and retention limit.
- Put noisy, high-volume data (logs, databases, container runtime) on a separate volume so that one consumer cannot fill the root filesystem.
- Add monitoring and alerts on both usage percentage and predicted time to full, for blocks and inodes.
- Fix the application behavior that causes excessive log output, such as an error that is logged in a tight loop.
- If capacity is genuinely insufficient, extend the volume and the filesystem. **Changes state:** take a snapshot or backup first.

## Verification

All commands in this section are **read-only**.

1. The filesystem has sufficient free space and inodes, and usage is back below the alert threshold.

   ```bash
   df -hT
   df -i
   ```

2. No process holds deleted files.

   ```bash
   sudo lsof +L1
   ```

3. The affected services run normally and logs show no further `No space left on device` errors.

   ```bash
   sudo journalctl -u <service> --since "30 min ago" --no-pager | tail -n 50
   ```

4. Kubernetes: the node is `Ready`, `DiskPressure` is `False`, and pods scheduled to the node start and run.

   ```bash
   kubectl describe node <node>
   kubectl get pods -A -o wide --field-selector spec.nodeName=<node>
   ```

5. Observe the fill rate for at least one full rotation cycle to confirm that the root cause is fixed and usage does not climb again.
