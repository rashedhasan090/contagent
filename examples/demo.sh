#!/usr/bin/env bash
# End-to-end demo. Runs offline with the mock backend, no API key needed.
set -euo pipefail
cd "$(dirname "$0")"
S=support_desk.toml

echo "== validate =="
contagent validate "$S"
echo
echo "== forecast (1000 runs) =="
contagent run "$S" -n 1000
echo
echo "== what if the ops agent gets a tool gate? =="
contagent run "$S" -n 1000 -w add:tool_gate@ops | tail -4
echo
echo "== one episode that ends in impact =="
contagent trace "$S" --find impact
echo
echo "== rank single fixes =="
contagent rank "$S" -n 400 --top 8
echo
echo "== greedy plan with a budget of 4 =="
contagent plan "$S" -n 400 --budget 4
