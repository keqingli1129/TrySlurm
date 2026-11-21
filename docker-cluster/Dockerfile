# One image for every node of the practice cluster: Slurm + munge + CPU-only PyTorch.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      slurm-wlm munge python3 python3-venv iproute2 procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Every node is built from this image, so they all share the same munge key (fine for a lab).
RUN (id slurm >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin slurm) \
 && ([ -s /etc/munge/munge.key ] || mungekey --create --force) \
 && chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key

COPY slurm.conf /etc/slurm/slurm.conf
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH
WORKDIR /lab
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
