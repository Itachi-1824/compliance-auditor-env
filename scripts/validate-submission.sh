#!/usr/bin/env bash
# Pre-submission validation for EU AI Act Compliance Auditor
# Usage: bash scripts/validate-submission.sh [SPACE_URL]

set -e

SPACE_URL="${1:-https://itachi1824-compliance-auditor-env.hf.space}"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }

echo -e "${BOLD}Pre-submission validation${NC}"
echo "Space: $SPACE_URL"
echo ""

# 1. Required files
echo -e "${BOLD}Step 1/5: Required files${NC}"
[ -f "inference.py" ]   && pass "inference.py" || fail "inference.py missing"
[ -f "Dockerfile" ]     && pass "Dockerfile"   || fail "Dockerfile missing"
[ -f "openenv.yaml" ]   && pass "openenv.yaml" || fail "openenv.yaml missing"
[ -f "README.md" ]      && pass "README.md"    || fail "README.md missing"
[ -f "pyproject.toml" ] && pass "pyproject.toml" || fail "pyproject.toml missing"
[ -f "models.py" ]      && pass "models.py"    || fail "models.py missing"
[ -f "client.py" ]      && pass "client.py"    || fail "client.py missing"
[ -d "server" ]         && pass "server/"      || fail "server/ missing"

# 2. openenv validate
echo ""
echo -e "${BOLD}Step 2/5: openenv validate${NC}"
if command -v openenv &>/dev/null || python -m openenv.cli validate &>/dev/null; then
    python -m openenv.cli validate 2>&1 | grep -q "OK" && pass "openenv validate" || fail "openenv validate failed"
else
    echo "  ⚠ openenv CLI not installed, skipping"
fi

# 3. Space health
echo ""
echo -e "${BOLD}Step 3/5: Space health${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SPACE_URL/health")
[ "$HTTP_CODE" = "200" ] && pass "/health returns 200" || fail "/health returned $HTTP_CODE"

TASKS=$(curl -s "$SPACE_URL/tasks" | python -c "import sys,json; print(len(json.load(sys.stdin)['tasks']))" 2>/dev/null)
[ "$TASKS" -ge 3 ] && pass "/tasks returns $TASKS tasks (>=3)" || fail "Only $TASKS tasks (need 3+)"

# 4. Reset + Step
echo ""
echo -e "${BOLD}Step 4/5: Reset + API${NC}"
RESET=$(curl -s -X POST "$SPACE_URL/api/reset" -H "Content-Type: application/json" -d '{"difficulty":"easy"}')
SID=$(echo "$RESET" | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null)
[ -n "$SID" ] && pass "/api/reset creates session" || fail "/api/reset failed"

TOOL=$(curl -s -X POST "$SPACE_URL/api/call_tool" -H "Content-Type: application/json" -d "{\"session_id\":\"$SID\",\"tool_name\":\"get_system_overview\",\"arguments\":{}}")
echo "$TOOL" | python -c "import sys,json; d=json.load(sys.stdin); assert 'result' in d" 2>/dev/null && pass "/api/call_tool works" || fail "/api/call_tool failed"

# 5. Stdout format check
echo ""
echo -e "${BOLD}Step 5/5: Stdout format${NC}"
grep -q "API_BASE_URL" inference.py && pass "API_BASE_URL in inference.py" || fail "Missing API_BASE_URL"
grep -q "MODEL_NAME" inference.py && pass "MODEL_NAME in inference.py" || fail "Missing MODEL_NAME"
grep -q "HF_TOKEN" inference.py && pass "HF_TOKEN in inference.py" || fail "Missing HF_TOKEN"
grep -q '\[START\]' inference.py && pass "[START] format present" || fail "Missing [START]"
grep -q '\[STEP\]' inference.py && pass "[STEP] format present" || fail "Missing [STEP]"
grep -q '\[END\]' inference.py && pass "[END] format present" || fail "Missing [END]"

echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${GREEN}${BOLD}  All checks passed!${NC}"
echo -e "${GREEN}${BOLD}  Ready to submit.${NC}"
echo -e "${BOLD}========================================${NC}"
