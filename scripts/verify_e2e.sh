#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "[1/6] Health"
curl -fsS "$BASE_URL/api/v1/health" | cat

echo

echo "[2/6] Dependency preflight"
curl -fsS "$BASE_URL/health/dependencies" | cat

echo

echo "[3/6] Create sample text"
cat > /tmp/e2e_text.txt << 'EOF'
John Smith from Acme Corp invested $5000 on 15 January 2024.
The meeting was cancelled due to budget cuts, but it will be rescheduled.
EOF

echo "[4/6] Analyze text upload"
ANALYZE_JSON=$(curl -fsS -X POST "$BASE_URL/api/v1/documents/upload" -F "file=@/tmp/e2e_text.txt")
echo "$ANALYZE_JSON" | python3 -m json.tool | head -n 40
DOC_ID=$(echo "$ANALYZE_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["document_id"])')

echo "[5/6] Fetch by document id: $DOC_ID"
curl -fsS "$BASE_URL/api/v1/documents/results/$DOC_ID" | python3 -m json.tool | head -n 30

echo "[6/6] Metrics"
curl -fsS "$BASE_URL/api/v1/metrics" | cat

echo
echo "E2E verification complete."
