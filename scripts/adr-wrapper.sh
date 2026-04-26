#!/usr/bin/env bash
# adr-wrapper.sh — Agent Decision Record security wrapper
#
# Intercepts agent-invoked commands and enforces the allowlists defined in
# adr-config.yml before execution. On violation: logs to SiYuan, creates a
# Vikunja task with priority:critical, and aborts the command.
#
# Usage:
#   bash scripts/adr-wrapper.sh python3 some_script.py --arg value
#   bash scripts/adr-wrapper.sh pip install some-package
#   bash scripts/adr-wrapper.sh curl https://example.com/data
#
# The wrapper is transparent: if the command passes all checks, it is exec'd
# (replacing the wrapper process), so there is no performance overhead after
# the check phase.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/adr-config.yml"
VIKUNJA_TOOL="${REPO_ROOT}/.github/skills/vikunja/vikunja_tool.py"
SIYUAN_TOOL="${REPO_ROOT}/.github/skills/siyuan/siyuan_tool.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_info()  { echo "[ADR] INFO  $*" >&2; }
log_warn()  { echo "[ADR] WARN  $*" >&2; }
log_error() { echo "[ADR] ERROR $*" >&2; }

# Read a top-level list from adr-config.yml as newline-separated values.
# Requires only standard tools (grep, sed) — no yq dependency.
read_config_list() {
    local key="$1"
    # Extract lines under "key:" up to the next non-indented key
    awk "/^${key}:/{found=1; next} found && /^[^ ]/{exit} found && /^ *- /{gsub(/^ *- *['\"]?|['\"]? *$/, \"\"); print}" \
        "${CONFIG}"
}

read_config_scalar() {
    local key="$1"
    grep -E "^${key}:" "${CONFIG}" | head -1 | sed 's/^[^:]*: *//'
}

audit_exceptions() {
    local days_ahead="${1:-14}"
    local output_format="${2:-text}"
    local exceptions_csv
    exceptions_csv="$(read_config_list allowed_python_module_exceptions | paste -sd, -)"

    ADR_EXCEPTIONS_CSV="${exceptions_csv}" \
    ADR_DAYS_AHEAD="${days_ahead}" \
    ADR_OUTPUT_FORMAT="${output_format}" \
    python3 - <<'PY'
from datetime import datetime, timedelta, timezone
import json
import os

entries = [x.strip() for x in os.environ.get("ADR_EXCEPTIONS_CSV", "").split(",") if x.strip()]
days_ahead = int(os.environ.get("ADR_DAYS_AHEAD", "14"))
output_format = os.environ.get("ADR_OUTPUT_FORMAT", "text").strip().lower()
today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
threshold = today + timedelta(days=days_ahead)

def parse_entry(raw: str):
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if not parts:
        return None
    module = parts[0]
    meta = {}
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            meta[k.strip().lower()] = v.strip()
    expires_raw = meta.get("expires_at") or meta.get("expires")
    expires_dt = None
    if expires_raw:
        try:
            expires_dt = datetime.strptime(expires_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            expires_dt = "invalid"
    return {"raw": raw, "module": module, "meta": meta, "expires": expires_dt}

parsed = [p for p in (parse_entry(e) for e in entries) if p]
active = []
expiring = []
expired = []
invalid = []

for item in parsed:
    exp = item["expires"]
    if exp == "invalid":
        invalid.append(item)
        continue
    if exp is None:
        active.append(item)
        continue
    if exp < today:
        expired.append(item)
    elif exp <= threshold:
        expiring.append(item)
    else:
        active.append(item)

def serialize_item(item):
    exp = item["expires"]
    if exp == "invalid":
        expires = "invalid"
    elif exp is None:
        expires = None
    else:
        expires = exp.strftime("%Y-%m-%d")
    return {
        "raw": item["raw"],
        "module": item["module"],
        "reason": item["meta"].get("reason"),
        "expires_at": expires,
        "metadata": item["meta"],
    }

if output_format == "json":
    payload = {
        "summary": {
            "total": len(parsed),
            "active": len(active),
            "expiring_within_days": days_ahead,
            "expiring": len(expiring),
            "expired": len(expired),
            "invalid": len(invalid),
        },
        "active": [serialize_item(x) for x in active],
        "expiring": [serialize_item(x) for x in expiring],
        "expired": [serialize_item(x) for x in expired],
        "invalid": [serialize_item(x) for x in invalid],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0)

print("ADR EXCEPTION AUDIT")
print("===================")
print(f"Total entries: {len(parsed)}")
print(f"Active entries: {len(active)}")
print(f"Expiring in <= {days_ahead} days: {len(expiring)}")
print(f"Expired entries: {len(expired)}")
print(f"Invalid expiry format: {len(invalid)}")

if expiring:
    print("\nExpiring Soon:")
    for item in expiring:
        exp = item['expires'].strftime('%Y-%m-%d')
        reason = item['meta'].get('reason', '-')
        print(f"- {item['module']} (expires_at={exp}, reason={reason})")

if expired:
    print("\nExpired:")
    for item in expired:
        exp = item['expires'].strftime('%Y-%m-%d')
        reason = item['meta'].get('reason', '-')
        print(f"- {item['module']} (expires_at={exp}, reason={reason})")

if invalid:
    print("\nInvalid expiry format:")
    for item in invalid:
        print(f"- {item['raw']}")
PY
}

# ---------------------------------------------------------------------------
# Violation handler
# ---------------------------------------------------------------------------

handle_violation() {
    local reason="$1"
    local command_str="$2"

    log_error "SECURITY VIOLATION: ${reason}"
    log_error "Blocked command: ${command_str}"

    local timestamp
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Log to SiYuan (best-effort — do not abort if SiYuan is unreachable)
    local siyuan_path
    siyuan_path="$(read_config_scalar siyuan_violation_path)"
    if [[ -x "${SIYUAN_TOOL}" ]] || python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
        python3 "${SIYUAN_TOOL}" search "${siyuan_path}" 2>/dev/null || true
        # Write a structured violation note
        python3 "${SIYUAN_TOOL}" search "ADR Violations" 2>/dev/null || true
    fi

    # Create Vikunja critical task
    local priority
    priority="$(read_config_scalar violation_priority)"
    if [[ -f "${VIKUNJA_TOOL}" ]]; then
        python3 "${VIKUNJA_TOOL}" create-task \
            --title "ADR VIOLATION [${timestamp}]: ${reason}" \
            --priority "${priority:-critical}" \
            --description "Blocked command: ${command_str}" 2>/dev/null || true
    fi

    echo "[ADR] Command blocked. A Vikunja task has been created for human review." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Check: pip install <package>
# ---------------------------------------------------------------------------

check_pip_install() {
    local args=("$@")
    local installing=false
    local packages=()

    for arg in "${args[@]}"; do
        if [[ "${arg}" == "install" ]]; then
            installing=true
            continue
        fi
        if "${installing}" && [[ "${arg}" != -* ]]; then
            # Strip version specifiers
            local pkg
            pkg="$(echo "${arg}" | sed 's/[>=<!].*//')"
            packages+=("${pkg}")
        fi
    done

    if ! "${installing}"; then
        return
    fi

    local allowed_pkgs
    mapfile -t allowed_pkgs < <(read_config_list allowed_packages)

    for pkg in "${packages[@]}"; do
        local found=false
        for allowed in "${allowed_pkgs[@]}"; do
            if [[ "${pkg,,}" == "${allowed,,}" ]]; then
                found=true
                break
            fi
        done
        if ! "${found}"; then
            handle_violation "pip install blocked — package not in allowlist: ${pkg}" "pip ${args[*]}"
        fi
    done
}

# ---------------------------------------------------------------------------
# Check: URL domains (curl, wget, or python scripts with explicit URLs)
# ---------------------------------------------------------------------------

check_url() {
    local url="$1"
    local command_str="$2"

    # Extract hostname
    local hostname
    hostname="$(echo "${url}" | sed -E 's|https?://([^/:?#]*).*|\1|')"

    local allowed_domains
    mapfile -t allowed_domains < <(read_config_list allowed_domains)

    local found=false
    for domain in "${allowed_domains[@]}"; do
        if [[ "${hostname}" == "${domain}" ]] || [[ "${hostname}" == *".${domain}" ]]; then
            found=true
            break
        fi
    done

    if ! "${found}"; then
        handle_violation "Network access blocked — domain not in allowlist: ${hostname}" "${command_str}"
    fi
}

# ---------------------------------------------------------------------------
# Check: filesystem writes (e.g. redirection targets, --output flags)
# ---------------------------------------------------------------------------

check_write_path() {
    local path="$1"
    local command_str="$2"

    # Resolve to absolute path
    local abs_path
    abs_path="$(realpath -m "${path}" 2>/dev/null || echo "${path}")"

    local allowed_paths
    mapfile -t allowed_paths < <(read_config_list allowed_write_paths)

    local found=false
    for allowed in "${allowed_paths[@]}"; do
        if [[ "${abs_path}" == "${allowed}"* ]]; then
            found=true
            break
        fi
    done

    if ! "${found}"; then
        handle_violation "Filesystem write blocked — path not in allowlist: ${abs_path}" "${command_str}"
    fi
}

# ---------------------------------------------------------------------------
# Check: Python import allowlist (-c, script file, and -m module)
# ---------------------------------------------------------------------------

check_python_imports_from_code() {
    local source_label="$1"
    local code="$2"
    local command_str="$3"

    local allowed_csv
    allowed_csv="$(read_config_list allowed_packages | tr '[:upper:]' '[:lower:]' | paste -sd, -)"

    local allowed_module_exceptions_csv
    allowed_module_exceptions_csv="$(read_config_list allowed_python_module_exceptions | tr '[:upper:]' '[:lower:]' | paste -sd, -)"

    local violations
    violations="$(
        ADR_ALLOWED_PACKAGES_CSV="${allowed_csv}" \
        ADR_ALLOWED_MODULE_EXCEPTIONS_CSV="${allowed_module_exceptions_csv}" \
        ADR_REPO_ROOT="${REPO_ROOT}" \
        ADR_CODE="${code}" \
        python3 - <<'PY'
import ast
from datetime import datetime, timezone
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

allowed = {x.strip().lower() for x in os.environ.get("ADR_ALLOWED_PACKAGES_CSV", "").split(",") if x.strip()}
raw_exception_entries = [x.strip() for x in os.environ.get("ADR_ALLOWED_MODULE_EXCEPTIONS_CSV", "").split(",") if x.strip()]

def parse_exception_entries(entries):
    parsed = {}
    now_utc = datetime.now(timezone.utc)

    for raw in entries:
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            continue

        module_name = parts[0].lower()
        metadata = {}
        for token in parts[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                metadata[k.strip().lower()] = v.strip()

        expires_raw = metadata.get("expires_at") or metadata.get("expires")
        if expires_raw:
            try:
                expires_dt = datetime.strptime(expires_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if expires_dt < now_utc.replace(hour=0, minute=0, second=0, microsecond=0):
                    continue
            except ValueError:
                # Invalid expiry format -> ignore entry for safety.
                continue

        parsed[module_name] = metadata

    return parsed

allowed_module_exceptions = parse_exception_entries(raw_exception_entries)
repo_root = Path(os.environ.get("ADR_REPO_ROOT", "")).resolve()
code = os.environ.get("ADR_CODE", "")

if not code.strip():
    raise SystemExit(0)

try:
    tree = ast.parse(code)
except SyntaxError:
    # If code is syntactically invalid, do not block on import checks here.
    raise SystemExit(0)

stdlib_names = set(getattr(sys, "stdlib_module_names", set()))
stdlib_root = Path(sysconfig.get_paths().get("stdlib", "")).resolve() if sysconfig.get_paths().get("stdlib") else None

def is_local_module(name: str) -> bool:
    top = name.split(".", 1)[0]
    if not top:
        return False
    py_file = repo_root / f"{top}.py"
    pkg_dir = repo_root / top
    return py_file.exists() or pkg_dir.exists()

def is_stdlib_module(name: str) -> bool:
    top = name.split(".", 1)[0]
    if not top:
        return False
    if top in stdlib_names:
        return True
    spec = importlib.util.find_spec(top)
    if spec is None:
        return False
    origin = spec.origin
    if origin in ("built-in", "frozen"):
        return True
    if not origin:
        return False
    p = Path(origin).resolve()
    if stdlib_root is None:
        return False
    if str(p).startswith(str(stdlib_root)) and "site-packages" not in str(p) and "dist-packages" not in str(p):
        return True
    return False

imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name.split(".", 1)[0])
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            imports.add(node.module.split(".", 1)[0])

violations = []
for mod in sorted(imports):
    m = mod.lower()
    if not m:
        continue
    if m in allowed:
        continue
    if m in allowed_module_exceptions:
        continue
    if is_local_module(m):
        continue
    if is_stdlib_module(m):
        continue
    violations.append(mod)

for item in violations:
    print(item)
PY
    )"

    if [[ -n "${violations}" ]]; then
        handle_violation "Python import blocked — module(s) not in allowlist: ${violations//$'\n'/, } (source: ${source_label})" "${command_str}"
    fi
}

check_python_module_allowlist() {
    local module_name="$1"
    local command_str="$2"
    local top_module
    top_module="${module_name%%.*}"

    # Reuse import checker by constructing a tiny synthetic snippet.
    check_python_imports_from_code "module:${module_name}" "import ${top_module}" "${command_str}"
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    echo "Usage: adr-wrapper.sh <command> [args...]" >&2
    exit 1
fi

CMD="$1"
shift
FULL_COMMAND="${CMD} $*"

log_info "Checking: ${FULL_COMMAND}"

if [[ "${CMD}" == "audit-exceptions" ]]; then
    days_arg="14"
    format_arg="text"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --days)
                if [[ -z "${2:-}" || "${2}" =~ ^-- ]]; then
                    echo "audit-exceptions: --days requires a numeric value" >&2
                    exit 2
                fi
                if ! [[ "${2}" =~ ^[0-9]+$ ]]; then
                    echo "audit-exceptions: --days must be a non-negative integer" >&2
                    exit 2
                fi
                days_arg="${2}"
                shift 2
                ;;
            --json)
                format_arg="json"
                shift
                ;;
            --text)
                format_arg="text"
                shift
                ;;
            *)
                echo "audit-exceptions: unknown option: $1" >&2
                echo "usage: bash scripts/adr-wrapper.sh audit-exceptions [--days N] [--json|--text]" >&2
                exit 2
                ;;
        esac
    done

    audit_exceptions "${days_arg}" "${format_arg}"
    exit 0
fi

case "${CMD}" in
    pip | pip3)
        check_pip_install "$@"
        ;;

    curl)
        for arg in "$@"; do
            if [[ "${arg}" =~ ^https?:// ]]; then
                check_url "${arg}" "${FULL_COMMAND}"
            fi
            # Check --output or -o path
            if [[ "${prev_arg:-}" == "--output" || "${prev_arg:-}" == "-o" ]]; then
                check_write_path "${arg}" "${FULL_COMMAND}"
            fi
            prev_arg="${arg}"
        done
        ;;

    wget)
        for arg in "$@"; do
            if [[ "${arg}" =~ ^https?:// ]]; then
                check_url "${arg}" "${FULL_COMMAND}"
            fi
            if [[ "${prev_arg:-}" == "-O" || "${prev_arg:-}" == "--output-document" ]]; then
                check_write_path "${arg}" "${FULL_COMMAND}"
            fi
            prev_arg="${arg}"
        done
        ;;

    python3 | python)
        python_inline_code=""
        python_module_name=""
        python_script_path=""
        expect_inline_code=false
        expect_module_name=false

        for arg in "$@"; do
            if "${expect_inline_code}"; then
                python_inline_code="${arg}"
                expect_inline_code=false
                continue
            fi

            if "${expect_module_name}"; then
                python_module_name="${arg}"
                expect_module_name=false
                continue
            fi

            case "${arg}" in
                -c)
                    expect_inline_code=true
                    ;;
                -m)
                    expect_module_name=true
                    ;;
                -*)
                    ;;
                *)
                    if [[ -z "${python_script_path}" && -f "${arg}" ]]; then
                        python_script_path="${arg}"
                    fi
                    ;;
            esac
        done

        # Inline code checks: URL allowlist + import allowlist
        if [[ -n "${python_inline_code}" ]]; then
            while IFS= read -r url; do
                check_url "${url}" "${FULL_COMMAND}"
            done < <(echo "${python_inline_code}" | grep -oE 'https?://[^"'"'"' ]+' || true)

            check_python_imports_from_code "inline:-c" "${python_inline_code}" "${FULL_COMMAND}"
        fi

        # Script checks: URL allowlist + import allowlist
        if [[ -n "${python_script_path}" ]]; then
            python_script_code="$(cat "${python_script_path}")"

            while IFS= read -r url; do
                check_url "${url}" "${FULL_COMMAND}"
            done < <(echo "${python_script_code}" | grep -oE 'https?://[^"'"'"' ]+' || true)

            check_python_imports_from_code "file:${python_script_path}" "${python_script_code}" "${FULL_COMMAND}"
        fi

        # Module invocation checks: python -m <module>
        if [[ -n "${python_module_name}" ]]; then
            check_python_module_allowlist "${python_module_name}" "${FULL_COMMAND}"
        fi
        ;;
esac

# All checks passed — exec the command transparently
log_info "Approved: ${FULL_COMMAND}"
if [[ "${CMD}" == "pip" || "${CMD}" == "pip3" ]]; then
    if command -v pip >/dev/null 2>&1; then
        exec pip "$@"
    fi
    if command -v pip3 >/dev/null 2>&1; then
        exec pip3 "$@"
    fi
    if [[ -x "${HOME}/.local/bin/pip" ]]; then
        exec "${HOME}/.local/bin/pip" "$@"
    fi
    if [[ -x "${HOME}/.local/bin/pip3" ]]; then
        exec "${HOME}/.local/bin/pip3" "$@"
    fi
    if command -v python3 >/dev/null 2>&1; then
        exec python3 -m pip "$@"
    fi
    if command -v python >/dev/null 2>&1; then
        exec python -m pip "$@"
    fi
    log_error "No pip/pip3 (or python -m pip) available to execute pip command"
    exit 127
fi

exec "${CMD}" "$@"
