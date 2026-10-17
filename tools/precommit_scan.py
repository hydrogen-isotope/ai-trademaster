"""
Blocks secrets, licensed market data and personal details from entering git or reaching GitHub.

  python tools/precommit_scan.py              staged files (pre-commit hook)
  python tools/precommit_scan.py --pre-push   every commit about to be pushed; reads git's pre-push stdin (pre-push hook)
  python tools/precommit_scan.py --history    every version of every file in all history + commit metadata (CI)
  python tools/precommit_scan.py --all        files in HEAD

Checks:
  - blocked paths: private/ data/ models/ reports/, data/model file types, credential files
  - binaries and files over 2 MB
  - secrets: private keys, GitHub/AWS/Google/Slack/API tokens, hard-coded passwords, IBKR-style account numbers
  - licensed market data: long numeric series, $ prices, ticker+price pairs, real-time unix timestamps, Yahoo payloads
  - personal details: e-mail addresses (except no-reply), Windows user paths, PC hostnames
  - commit metadata (push/history modes): author and committer e-mails must be no-reply addresses

Exit code 1 lists every problem. A line can be exempted with the marker  precommit-allow  (use sparingly,
reviewed). Do not weaken the patterns without review.
"""
import re, subprocess, sys

MAX_BYTES = 2 * 1024 * 1024
ZERO = "0" * 40

BLOCKED_PATHS = [
    (re.compile(r"^(private|data|models|reports)/"), "private/ data, model weights and reports must stay local (licensing)"),
    (re.compile(r"\.(tmrp|parquet|feather|dbn|pcap|onnx|jsonl|npy|npz|pt|pth|pkl|h5)(\.zst)?$", re.I), "data or model artifact"),
    (re.compile(r"(^|/)\.env(\.|$)|\.(pfx|p12|pem|key|secret)$", re.I), "secret/credential file"),
    (re.compile(r"(^|/)(ibkr|live)[^/]*\.json$", re.I), "broker/live configuration (keep local)"),
]

CONTENT_PATTERNS = [
    # secrets
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "Google API key"),
    (re.compile(r"\bxox[abpr]-[0-9A-Za-z-]{10,}\b"), "Slack token"),
    (re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{20,}\b"), "API secret key"),
    (re.compile(r"""(?ix)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)\b
                    \s*[:=]\s*["'][^"'\s]{6,}["']"""), "hard-coded credential assignment"),
    (re.compile(r"(?i)\b(U|DU|DF)\d{6,8}\b"), "possible IBKR account number"),
    # licensed market data
    (re.compile(r"(?:-?\d+\.\d+\s*[,;|\s]\s*){20,}"), "long numeric series (possible price/volume data)"),
    (re.compile(r"\$\s?\d{2,5}\.\d{2}\b"), "dollar price"),
    (re.compile(r"\b(MU|NVDA|AMD|AAPL|MSFT|GOOGL|AMZN|META|TSLA|SMH|QQQ|SPY|AVGO|INTC|QCOM)\b[^\n]{0,20}\b\d{2,4}\.\d{2,4}\b"),
     "ticker next to a price"),
    (re.compile(r"\b1[67]\d{8}\b"), "real-time unix timestamp (possible market data)"),
    (re.compile(r'"(regularMarketPrice|chartPreviousClose|currentTradingPeriod)"'), "Yahoo market-data payload"),
    # personal details
    (re.compile(r"\b[\w.+-]+@(?!users\.noreply\.github\.com\b|anthropic\.com\b)[\w-]+(\.[\w-]+)*\.[a-z]{2,}\b", re.I),
     "e-mail address (use a no-reply address)"),
    (re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\s\"']+", re.I), "Windows user path"),
    (re.compile(r"\bDESKTOP-[A-Z0-9]{5,}\b"), "PC hostname"),
]
# Version pins such as "lightweight-charts@4.2.0" look like e-mails; they are fine.
EMAIL_FALSE_POSITIVE = re.compile(r"@\d+\.\d+")
NOREPLY = re.compile(r"(@users\.noreply\.github\.com|^noreply@anthropic\.com)$", re.I)
TEXT_EXT = re.compile(r"\.(cs|csproj|props|py|ps1|cmd|sh|json|md|txt|yml|yaml|html|js|css|slnx|toml|ini|cfg)$|(^|/)(LICENSE|pre-commit|pre-push|\.gitignore|\.gitattributes)$", re.I)


def git(*args, binary=False, stdin=None):
    r = subprocess.run(["git", *args], capture_output=True, check=True, input=stdin)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def scan_blob(path, data, problems, where=""):
    tag = f"{path}{where}"
    for rx, why in BLOCKED_PATHS:
        if rx.search(path):
            problems.append(f"{tag}: blocked path - {why}")
    if len(data) > MAX_BYTES:
        problems.append(f"{tag}: {len(data) / 1e6:.1f} MB exceeds {MAX_BYTES / 1e6:.0f} MB (data/binaries stay out of git)")
    if b"\x00" in data[:8000]:
        problems.append(f"{tag}: binary file")
        return
    if not TEXT_EXT.search(path):
        return
    text = data.decode("utf-8", errors="replace")
    for n, line in enumerate(text.splitlines(), 1):
        if "precommit-allow" in line:
            continue
        for rx, why in CONTENT_PATTERNS:
            for m in rx.finditer(line):
                if why.startswith("e-mail") and EMAIL_FALSE_POSITIVE.search(m.group(0)):
                    continue
                problems.append(f"{tag}:{n}: {why}: {m.group(0)[:60]!r}")
                break


def scan_commits(commits, problems):
    """Every blob introduced by these commits (all versions, not just the tip) plus their metadata."""
    seen = set()
    for c in commits:
        meta = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", c).split("\x00")
        for kind, email in (("author", meta[1]), ("committer", meta[3])):
            if not NOREPLY.search(email.strip()):
                problems.append(f"commit {c[:8]}: {kind} e-mail {email!r} is not a no-reply address "
                                f"(git config user.email <id>+<login>@users.noreply.github.com)")
        for m in re.finditer(r"[\w.+-]+@[\w-]+(\.[\w-]+)*\.[a-z]{2,}", meta[4], re.I):
            if not NOREPLY.search(m.group(0)):
                problems.append(f"commit {c[:8]}: e-mail in commit message: {m.group(0)!r}")
        parents = git("rev-list", "--parents", "-n", "1", c).split()[1:]
        base = parents[0] if parents else git("hash-object", "-t", "tree", "--stdin", stdin=b"").strip()
        diff = git("diff-tree", "-r", "--no-commit-id", "--diff-filter=AM", "--name-only", "-z", base, c)
        for path in filter(None, diff.split("\x00")):
            sha = git("rev-parse", f"{c}:{path}").strip()
            if sha in seen:
                continue
            seen.add(sha)
            scan_blob(path, git("cat-file", "-p", sha, binary=True), problems, f" (commit {c[:8]})")


def main():
    problems, what = [], ""
    if "--pre-push" in sys.argv:
        commits = []
        for line in sys.stdin.read().splitlines():
            parts = line.split()
            if len(parts) != 4 or parts[1] == ZERO:      # branch deletion: nothing to scan
                continue
            local, remote = parts[1], parts[3]
            rng = [local, "--not", "--remotes"] if remote == ZERO else [f"{remote}..{local}"]
            commits += git("rev-list", "--reverse", *rng).split()
        scan_commits(dict.fromkeys(commits), problems)
        what = f"{len(commits)} commit(s) to push"
    elif "--history" in sys.argv:
        commits = git("rev-list", "--reverse", "--all").split()
        scan_commits(commits, problems)
        what = f"all history: {len(commits)} commit(s)"
    else:
        files = git("ls-files") if "--all" in sys.argv else git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
        files = [f for f in files.splitlines() if f]
        for f in files:
            spec = f"HEAD:{f}" if "--all" in sys.argv else f":{f}"
            try:
                data = git("show", spec, binary=True)
            except subprocess.CalledProcessError:
                continue
            scan_blob(f, data, problems)
        what = f"{len(files)} file(s)"
    if problems:
        print(f"BLOCKED by tools/precommit_scan.py ({what}):")
        for p in problems:
            print("  - " + p)
        print("Fix the content, move secrets to environment variables / Windows Credential Manager, keep data and models "
              "under private/. If a pushed commit is affected, rewrite it locally before pushing (see SECURITY.md).")
        return 1
    print(f"precommit_scan: {what} clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
