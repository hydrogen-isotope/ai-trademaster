"""
Blocks commits that would leak secrets or licensed market data. Used by .githooks/pre-commit and CI.

  python tools/precommit_scan.py          scan files staged for commit
  python tools/precommit_scan.py --all    scan every tracked file (CI)

Exit code 1 lists every problem found. False positives: rename the variable or move the value to the environment;
do not weaken the patterns without review.
"""
import re, subprocess, sys

MAX_BYTES = 2 * 1024 * 1024

BLOCKED_PATHS = [
    (re.compile(r"^(private|data|models|reports)/"), "private/ data, model weights and reports must stay local (licensing)"),
    (re.compile(r"\.(tmrp|parquet|feather|dbn|pcap|onnx|jsonl)(\.zst)?$", re.I), "data or model artifact"),
    (re.compile(r"(^|/)\.env(\.|$)|\.(pfx|p12|pem|key|secret)$", re.I), "secret/credential file"),
    (re.compile(r"(^|/)(ibkr|live)[^/]*\.json$", re.I), "broker/live configuration (keep local)"),
]

SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "Google API key"),
    (re.compile(r"\bxox[abpr]-[0-9A-Za-z-]{10,}\b"), "Slack token"),
    (re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{20,}\b"), "API secret key"),
    (re.compile(r"""(?ix)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)\b
                    \s*[:=]\s*["'][^"'\s]{6,}["']"""), "hard-coded credential assignment"),
    (re.compile(r"(?i)\b(U|DU|DF)\d{6,8}\b"), "possible IBKR account number"),
]

TEXT_EXT = re.compile(r"\.(cs|csproj|props|py|ps1|cmd|sh|json|md|txt|yml|yaml|html|js|css|slnx|toml|ini|cfg)$", re.I)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def main():
    scan_all = "--all" in sys.argv
    files = git("ls-files") if scan_all else git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    files = [f for f in files.splitlines() if f]
    problems = []
    for f in files:
        for rx, why in BLOCKED_PATHS:
            if rx.search(f):
                problems.append(f"{f}: blocked path — {why}")
        try:
            blob = subprocess.run(["git", "show", f":{f}"] if not scan_all else ["git", "show", f"HEAD:{f}"],
                                  capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            try:
                blob = open(f, "rb").read()
            except OSError:
                continue
        if len(blob) > MAX_BYTES:
            problems.append(f"{f}: {len(blob) / 1e6:.1f} MB exceeds {MAX_BYTES / 1e6:.0f} MB (data/binaries stay out of git)")
        if not TEXT_EXT.search(f):
            continue
        text = blob.decode("utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if "precommit-allow" in line:
                continue
            for rx, why in SECRET_PATTERNS:
                if rx.search(line):
                    problems.append(f"{f}:{n}: {why}")
    if problems:
        print("COMMIT BLOCKED by tools/precommit_scan.py:")
        for p in problems:
            print("  - " + p)
        print("Fix the files (or move secrets to environment variables / Windows Credential Manager).")
        return 1
    print(f"precommit_scan: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
