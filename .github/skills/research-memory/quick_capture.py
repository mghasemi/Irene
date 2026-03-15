import argparse
import datetime as dt
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(__file__)
TOOL_PATH = os.path.join(SCRIPT_DIR, "research_memory_tool.py")


def _default_title(text, prefix):
    words = text.strip().split()
    short = " ".join(words[:8]).strip()
    if not short:
        short = "Untitled"
    if len(words) > 8:
        short += "..."
    return f"{prefix}: {short}"


def _run_tool(args):
    cmd = [sys.executable, TOOL_PATH] + args
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _append_tag(tags, tag):
    if not tags.strip():
        return tag
    return f"{tags},{tag}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Short commands for capturing and recalling research memory."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    idea = sub.add_parser("idea", help="Quickly capture an idea/observation")
    idea.add_argument("text", help="Observation text")
    idea.add_argument("--title", default="")
    idea.add_argument("--why", default="")
    idea.add_argument("--tags", default="")
    idea.add_argument("--context", default="")

    web = sub.add_parser("web", help="Quickly capture a web finding")
    web.add_argument("query", help="Exact web query used")
    web.add_argument("url", help="Source URL")
    web.add_argument("summary", help="Why this result is useful")
    web.add_argument("--title", default="")
    web.add_argument("--why", default="")
    web.add_argument("--tags", default="")
    web.add_argument("--context", default="")

    session = sub.add_parser("session", help="Capture a session summary note")
    session.add_argument("summary", help="Session summary")
    session.add_argument("--title", default="")
    session.add_argument("--tags", default="")
    session.add_argument("--context", default="session")

    find_cmd = sub.add_parser("find", help="Search saved memories")
    find_cmd.add_argument("query", help="Search query")
    find_cmd.add_argument("--kind", choices=["idea", "web"])
    find_cmd.add_argument("--limit", type=int, default=10)

    recent = sub.add_parser("recent", help="Show recent memories")
    recent.add_argument("--limit", type=int, default=10)

    return parser


def main():
    args = build_parser().parse_args()

    if args.cmd == "idea":
        title = args.title.strip() or _default_title(args.text, "Idea")
        cmd = [
            "add-idea",
            "--title",
            title,
            "--observation",
            args.text,
            "--tags",
            args.tags,
            "--context",
            args.context,
        ]
        if args.why.strip():
            cmd.extend(["--why", args.why])
        raise SystemExit(_run_tool(cmd))

    if args.cmd == "web":
        title = args.title.strip() or _default_title(args.query, "Web")
        cmd = [
            "add-web",
            "--query",
            args.query,
            "--url",
            args.url,
            "--title",
            title,
            "--summary",
            args.summary,
            "--tags",
            _append_tag(args.tags, "web"),
            "--context",
            args.context,
        ]
        if args.why.strip():
            cmd.extend(["--why", args.why])
        raise SystemExit(_run_tool(cmd))

    if args.cmd == "session":
        title = args.title.strip() or f"Session {dt.datetime.now().strftime('%Y-%m-%d')}"
        cmd = [
            "add-idea",
            "--title",
            title,
            "--observation",
            args.summary,
            "--tags",
            _append_tag(args.tags, "session"),
            "--context",
            args.context,
        ]
        raise SystemExit(_run_tool(cmd))

    if args.cmd == "find":
        cmd = ["search", "--query", args.query, "--limit", str(args.limit)]
        if args.kind:
            cmd.extend(["--kind", args.kind])
        raise SystemExit(_run_tool(cmd))

    if args.cmd == "recent":
        raise SystemExit(_run_tool(["recent", "--limit", str(args.limit)]))

    raise SystemExit(1)


if __name__ == "__main__":
    main()
