#!/bin/bash
#SBATCH --job-name=postgres
#SBATCH --output=logs/postgres/%A.out
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

echo "Running postgres on $HOSTNAME"

set -euo pipefail

STOREDIR="$PWD/postgres"
POSTGRES_IMAGE="$HOME/apps/apptainer/postgres.sif"

mkdir -p "$STOREDIR/data"
mkdir -p "$STOREDIR/run"
echo "*" > "$STOREDIR/.gitignore"  # optional

ENV_FILE="$STOREDIR/.env"
PGHOST_VALUE="$(hostname -f)"

if [[ ! -f "$ENV_FILE" ]]; then
    POSTGRES_USER="${POSTGRES_USER:-$(whoami)}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -hex 32)}"
    POSTGRES_DB="${POSTGRES_DB:-optuna}"
    PGPORT="${PGPORT:-5432}"

    cat > "$ENV_FILE" <<EOL
POSTGRES_USER="$POSTGRES_USER"
POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
POSTGRES_DB="$POSTGRES_DB"
PGHOST="$PGHOST_VALUE"
PGPORT="$PGPORT"
EOL
else
    if grep -q '^PGHOST=' "$ENV_FILE"; then
        sed -i "s|^PGHOST=.*|PGHOST=\"$PGHOST_VALUE\"|" "$ENV_FILE"
    else
        echo "PGHOST=\"$PGHOST_VALUE\"" >> "$ENV_FILE"
    fi
fi

if [[ "${1:-}" == "instance" ]]; then
    apptainer instance start --bind "$STOREDIR/data:/var/lib/postgresql,$STOREDIR/run:/var/run/postgresql" --env-file "$STOREDIR/.env" "$POSTGRES_IMAGE" postgres-server
    apptainer instance list
else
    apptainer run --bind "$STOREDIR/data:/var/lib/postgresql" --bind "$STOREDIR/run:/var/run/postgresql" --env-file "$STOREDIR/.env" "$POSTGRES_IMAGE"
fi
