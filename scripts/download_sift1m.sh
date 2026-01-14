#!/usr/bin/env bash
# Downloads SIFT1M into data/sift/ and checks the files.
set -euo pipefail

DATA_DIR="${1:-data}"
DEST="${DATA_DIR}/sift"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

FILES=(
  "sift_base.fvecs 516000000 128"
  "sift_query.fvecs 5160000 128"
  "sift_learn.fvecs 51600000 128"
  "sift_groundtruth.ivecs 4040000 100"
)

mkdir -p "${DEST}"

need_download=0
for entry in "${FILES[@]}"; do
  read -r name _ _ <<<"${entry}"
  [[ -f "${DEST}/${name}" ]] || need_download=1
done

if [[ "${need_download}" -eq 1 ]]; then
  tarball="${DATA_DIR}/sift.tar.gz"
  if [[ ! -f "${tarball}" ]]; then
    echo "Downloading ${URL}"
    curl --fail --retry 3 --progress-bar -o "${tarball}.part" "${URL}"
    mv "${tarball}.part" "${tarball}"
  fi
  tar -xzf "${tarball}" -C "${DATA_DIR}"
  rm -f "${tarball}"
fi

status=0
for entry in "${FILES[@]}"; do
  read -r name bytes dim <<<"${entry}"
  path="${DEST}/${name}"
  actual_bytes=$(stat -c %s "${path}")
  actual_dim=$(od -An -t d4 -N 4 "${path}" | tr -d ' ')
  if [[ "${actual_bytes}" != "${bytes}" || "${actual_dim}" != "${dim}" ]]; then
    echo "FAIL ${name}: bytes=${actual_bytes} (want ${bytes}), dim=${actual_dim} (want ${dim})"
    status=1
  else
    echo "OK   ${name}: ${actual_bytes} bytes, dim=${actual_dim}"
  fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"

sums="${DEST}/SHA256SUMS"
if [[ -f "${sums}" ]]; then
  (cd "${DEST}" && sha256sum --quiet -c SHA256SUMS) && echo "OK   SHA-256 matches recorded sums"
else
  (cd "${DEST}" && sha256sum sift_*.fvecs sift_*.ivecs > SHA256SUMS)
  echo "Recorded SHA-256 sums to ${sums}"
fi
