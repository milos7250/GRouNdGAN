FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y --no-install-recommends && apt-get install git g++ make && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /groundgan
COPY arboreto.patch requirements.txt ./

# Install dependencies from requirements.txt
RUN python -m pip install --no-cache-dir --root-user-action ignore -r /groundgan/requirements.txt && (rm -rf /root/.cache/pip || true)

# Patch arboreto bug
RUN patch /opt/conda/lib/python3.11/site-packages/arboreto/core.py /groundgan/arboreto.patch

# Run the main Python script
CMD ["python", "main.py", "--help"]