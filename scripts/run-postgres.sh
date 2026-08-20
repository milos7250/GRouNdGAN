#!/bin/bash
#SBATCH --job-name=postgres
#SBATCH --output=postgres.out
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

cat > "$STOREDIR/.env" <<EOL
POSTGRES_USER="$(whoami)"
POSTGRES_PASSWORD="TyBudesOratZa5KorunNaMojomPoli!"
POSTGRES_DB="optuna"
PGHOST="$(hostname -f)"
PGPORT="5432"
EOL

if [[ "${1:-}" == "instance" ]]; then
    apptainer instance start --bind "$STOREDIR/data:/var/lib/postgresql,$STOREDIR/run:/var/run/postgresql" --env-file "$STOREDIR/.env" "$POSTGRES_IMAGE" postgres-server
    apptainer instance list
else
    apptainer run --bind "$STOREDIR/data:/var/lib/postgresql" --bind "$STOREDIR/run:/var/run/postgresql" --env-file "$STOREDIR/.env" "$POSTGRES_IMAGE"
fi
