# Kubernetes ImagePullBackOff Troubleshooting

## Overview

`ErrImagePull` and `ImagePullBackOff` are pod statuses reported when the kubelet on a node cannot pull a container image. `ErrImagePull` is the initial failure. After repeated failures the kubelet waits with an exponential back-off between attempts (an increasing delay capped at 300 seconds, or 5 minutes) and the status changes to `ImagePullBackOff`. The pod stays in `Pending` with the container in `Waiting` state, so the application never starts.

The failure can be on any part of the path between the node and the image:

- the image reference is wrong: misspelled repository, tag that does not exist, or a digest that was removed
- authentication or authorization fails: missing, expired, or wrong `imagePullSecrets`, or a registry permission that was revoked
- the registry is unreachable from the node: DNS failure, network policy, proxy, firewall, TLS certificate problem, or a registry outage
- the registry rate-limits the node
- the node itself cannot store the image, for example because the disk is full or the container runtime is unhealthy
- the image architecture does not match the node (for example an `arm64` image on an `amd64` node)

This runbook assumes the pod has been scheduled to a node. For containers that start and then crash, use the Kubernetes CrashLoopBackOff runbook. For node disk problems, use the Linux disk-space exhaustion runbook.

Command conventions used in this runbook:

- **Read-only** commands only inspect state and are safe to run at any time.
- **Changes state** commands modify configuration, workloads, or cluster resources.
- **Destructive** commands delete or discard data and need an explicit check of the target first.

## Symptoms

- `kubectl get pods` shows `ErrImagePull` or `ImagePullBackOff`, and the pod never becomes ready.
- A Deployment rollout stalls with new pods in `Pending`/`ImagePullBackOff` while old pods keep serving.
- Pod events contain messages such as `Failed to pull image`, `manifest unknown`, `pull access denied`, `unauthorized`, `no such host`, `i/o timeout`, or `no space left on device`.
- Only some pods fail: pods on one node fail while pods on other nodes start normally. This points to a node-level cause rather than an image or credential problem.

## Diagnosis

All commands in this section are **read-only**. Replace `<namespace>`, `<pod>`, and `<node>` with the values for the affected workload.

1. Confirm the status and see which node the pod was scheduled to.

   ```bash
   kubectl get pods -n <namespace> -o wide
   ```

2. Read the events at the bottom of the pod description. The exact error text identifies the category of failure.

   ```bash
   kubectl describe pod <pod> -n <namespace>
   ```

3. Match the error message to a cause.

   | Message fragment | Likely cause |
   | ---------------- | ------------ |
   | `manifest unknown`, `not found` | Tag or digest does not exist, or the repository name is wrong |
   | `pull access denied`, `unauthorized`, `authentication required` | Missing, wrong, or expired pull credentials |
   | `no such host`, `i/o timeout`, `connection refused` | DNS, network, firewall, proxy, or registry outage |
   | `x509: certificate signed by unknown authority` | Registry TLS certificate not trusted by the node |
   | `toomanyrequests` | Registry rate limit reached |
   | `no space left on device` | Node disk is full |
   | `no matching manifest for linux/amd64` | Image does not support the node architecture |

4. Check the exact image reference in the workload, including registry host, repository, and tag or digest.

   ```bash
   kubectl get deployment <deployment> -n <namespace> \
     -o jsonpath='{.spec.template.spec.containers[*].image}'
   ```

   Avoid mutable tags such as `latest` in production workloads, because the same reference can resolve to different images over time.

5. Check which pull secrets the pod references and that they exist in the same namespace.

   ```bash
   kubectl get pod <pod> -n <namespace> -o jsonpath='{.spec.imagePullSecrets}'
   kubectl get secret -n <namespace> --field-selector type=kubernetes.io/dockerconfigjson
   ```

   Also check the service account, because pull secrets can be attached to it.

   ```bash
   kubectl get serviceaccount <service-account> -n <namespace> -o yaml
   ```

6. Verify the registry path from the same environment as the node, using shell access to the affected node. If you use a temporary debug pod instead, that creates a resource (**Changes state**) and should be removed afterwards. The checks themselves are read-only network checks; do not paste registry credentials into commands that are recorded in shell history or tickets.

   ```bash
   nslookup registry.example.com
   curl -sS -o /dev/null -w '%{http_code}\n' https://registry.example.com/v2/
   ```

   An HTTP `401` from the `/v2/` endpoint is normal for a registry that requires authentication and shows that the registry is reachable.

7. If the problem affects only one node, inspect the node conditions and the disk.

   ```bash
   kubectl describe node <node>
   ```

   Look for `DiskPressure`, `Ready`, and recent `ImageGCFailed`, `FreeDiskSpaceFailed`, or eviction events. On the node itself:

   ```bash
   df -h /var/lib/containerd /var/lib/kubelet
   crictl images
   ```

8. Check whether the failure started after a change: a new image tag, a rotated credential, a registry migration, or a network policy change.

## Resolution

Choose the branch that matches the cause found during diagnosis.

### Wrong image reference

1. Correct the repository or tag in the manifest in version control and apply it. **Changes state:** the update triggers a new rollout.

   ```bash
   kubectl apply -f deployment.yaml -n <namespace>
   ```

2. If the intended tag was never published, fix the build pipeline to publish it, or deploy a tag that exists. **Changes state.** Do not retag an existing image to hide a pipeline failure.
3. If a rollout is already stuck, roll back to the previous revision. **Changes state:**

   ```bash
   kubectl rollout undo deployment/<deployment> -n <namespace>
   ```

### Missing or expired pull credentials

1. Create or rotate the registry credential in the registry, using the identity provider or credential store of the organization. **Changes state:** this happens outside the cluster and may invalidate the old credential for other consumers. Never commit credentials to version control.
2. Re-create the pull secret from environment variables or a secret manager rather than typing values on the command line. **Changes state.** The variables below are placeholders and must be supplied by the operator from the credential store. This is an illustrative example, not a command to run as-is.

   ```bash
   kubectl create secret docker-registry registry-pull-secret \
     --docker-server=registry.example.com \
     --docker-username="$REGISTRY_USERNAME" \
     --docker-password="$REGISTRY_PASSWORD" \
     -n <namespace> --dry-run=client -o yaml | kubectl apply -f -
   ```

3. Make sure the pod or its service account references the secret in `imagePullSecrets`. **Changes state.**

   ```yaml
   spec:
     imagePullSecrets:
       - name: registry-pull-secret
   ```

4. Existing pods in `ImagePullBackOff` retry automatically, but the back-off can delay recovery by up to 5 minutes. Restart the rollout to speed this up. **Changes state:**

   ```bash
   kubectl rollout restart deployment/<deployment> -n <namespace>
   ```

### Registry unreachable or rate limited

1. Check the status page or monitoring of the registry and of the network path (DNS, proxy, firewall, network policy).
2. For a registry outage, wait for recovery or switch to a mirror or pull-through cache if one is configured. **Changes state:** switching means changing image references or runtime mirror configuration. Do not change production image references to an unapproved registry.
3. For rate limiting, authenticate the pulls with a credential that has a higher quota, or use a registry mirror. **Changes state.**

### Node cannot store the image

1. When the error is `no space left on device` or the node reports `DiskPressure`, follow the Linux disk-space exhaustion runbook for the node.
2. Removing unused images on the node frees space. **Destructive:** this deletes locally cached images that are not used by any container; they are pulled again when needed, which can add load to the registry. Confirm the node and review the list before removing anything.

   ```bash
   crictl rmi --prune
   ```

3. Cordon the node while it is repaired so no new pods are scheduled there. **Changes state:**

   ```bash
   kubectl cordon <node>
   ```

4. After the node is repaired, uncordon it. **Changes state:**

   ```bash
   kubectl uncordon <node>
   ```

### Architecture mismatch

Build and publish a multi-architecture image, or constrain the workload to nodes of the matching architecture with a `nodeSelector` on `kubernetes.io/arch`. **Changes state.**

## Verification

All commands in this section are **read-only**.

1. The pod moves from `Pending` to `Running` and all containers are ready.

   ```bash
   kubectl get pods -n <namespace> --watch
   ```

2. The pod events show `Pulled` (or `Already present on machine`) followed by `Created` and `Started`, with no new `Failed` or `BackOff` events.

   ```bash
   kubectl describe pod <pod> -n <namespace>
   ```

3. The rollout completes.

   ```bash
   kubectl rollout status deployment/<deployment> -n <namespace>
   ```

4. If a node was repaired, the node reports `Ready` without `DiskPressure`, and pods scheduled to it start normally.

   ```bash
   kubectl describe node <node>
   ```

5. Service-level indicators for the workload return to their normal range.
