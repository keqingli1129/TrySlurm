#!/bin/bash
# Starts munge, then either the controller or a compute daemon in the foreground.
set -e

mkdir -p /run/munge /var/spool/slurmctld /var/spool/slurmd
chown munge:munge /run/munge
chown slurm:slurm /var/spool/slurmctld

# --force: munged refuses to start otherwise because /var/log is group-writable in the Ubuntu image
runuser -u munge -- /usr/sbin/munged --force

case "${1:-}" in
  slurmctld)
    exec /usr/sbin/slurmctld -D
    ;;
  slurmd)
    echo "waiting for slurmctld..."
    for _ in $(seq 60); do
      scontrol ping 2>/dev/null | grep -q UP && break
      sleep 1
    done
    exec /usr/sbin/slurmd -D
    ;;
  *)
    exec "$@"
    ;;
esac
