import argparse
import datetime as dt
import json
import os
import sqlite3
import sys


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "research_memory.sqlite3")
DB_PATH = os.getenv("RESEARCH_MEMORY_DB", DEFAULT_DB_PATH)


class MemoryError(RuntimeError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL CHECK(kind IN ('idea', 'web')),
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            source_url TEXT NOT NULL DEFAULT '',
            query_text TEXT NOT NULL DEFAULT '',
            context TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED,
            title,
            content,
            tags,
            source_url,
            query_text,
            context
        )
        """
    )
    conn.commit()


def normalize_tags(raw):
    if not raw:
        return ""
    parts = []
    for chunk in raw.replace(";", ",").split(","):
        tag = chunk.strip().lower()
        if tag and tag not in parts:
            parts.append(tag)
    return ",".join(parts)


def row_to_dict(row):
    return {
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "content": row["content"],
        "tags": [t for t in row["tags"].split(",") if t],
        "source_url": row["source_url"],
        "query_text": row["query_text"],
        "context": row["context"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def refresh_fts(conn, memory_id):
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        conn.commit()
        return

    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    conn.execute(
        """
        INSERT INTO memories_fts (memory_id, title, content, tags, source_url, query_text, context)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            row["title"],
            row["content"],
            row["tags"],
            row["source_url"],
            row["query_text"],
            row["context"],
        ),
    )
    conn.commit()


def add_memory(conn, kind, title, content, tags="", source_url="", query_text="", context=""):
    now = utc_now()
    cur = conn.execute(
        """
        INSERT INTO memories (kind, title, content, tags, source_url, query_text, context, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (kind, title, content, normalize_tags(tags), source_url, query_text, context, now, now),
    )
    memory_id = cur.lastrowid
    conn.commit()
    refresh_fts(conn, memory_id)
    return get_memory(conn, memory_id)


def get_memory(conn, memory_id):
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        raise MemoryError(f"Memory id not found: {memory_id}")
    return row_to_dict(row)


def update_memory(conn, memory_id, title=None, content=None, tags=None, source_url=None, query_text=None, context=None):
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        raise MemoryError(f"Memory id not found: {memory_id}")

    next_title = title if title is not None else row["title"]
    next_content = content if content is not None else row["content"]
    next_tags = normalize_tags(tags) if tags is not None else row["tags"]
    next_source_url = source_url if source_url is not None else row["source_url"]
    next_query_text = query_text if query_text is not None else row["query_text"]
    next_context = context if context is not None else row["context"]
    now = utc_now()

    conn.execute(
        """
        UPDATE memories
        SET title = ?, content = ?, tags = ?, source_url = ?, query_text = ?, context = ?, updated_at = ?
        WHERE id = ?
        """,
        (next_title, next_content, next_tags, next_source_url, next_query_text, next_context, now, memory_id),
    )
    conn.commit()
    refresh_fts(conn, memory_id)
    return get_memory(conn, memory_id)


def delete_memory(conn, memory_id):
    exists = conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if exists is None:
        raise MemoryError(f"Memory id not found: {memory_id}")
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    conn.commit()
    return {"deleted_id": memory_id}


def recent_memories(conn, limit):
    rows = conn.execute(
        "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [row_to_dict(r) for r in rows]


def search_memories(conn, query, kind=None, limit=10):
    if not query.strip():
        raise MemoryError("Search query cannot be empty.")

    base_sql = """
        SELECT m.*
        FROM memories_fts f
        JOIN memories m ON m.id = f.memory_id
        WHERE f.memories_fts MATCH ?
    """
    params = [query]

    if kind:
        base_sql += " AND m.kind = ?"
        params.append(kind)

    base_sql += " ORDER BY bm25(memories_fts), m.updated_at DESC LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(base_sql, params).fetchall()
        return [row_to_dict(r) for r in rows]
    except sqlite3.OperationalError:
        like = f"%{query.lower()}%"
        sql = """
            SELECT *
            FROM memories
            WHERE (
                lower(title) LIKE ?
                OR lower(content) LIKE ?
                OR lower(tags) LIKE ?
                OR lower(source_url) LIKE ?
                OR lower(query_text) LIKE ?
                OR lower(context) LIKE ?
            )
        """
        p = [like, like, like, like, like, like]
        if kind:
            sql += " AND kind = ?"
            p.append(kind)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        p.append(limit)
        rows = conn.execute(sql, p).fetchall()
        return [row_to_dict(r) for r in rows]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tiny local memory for research ideas and web findings."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize local database")

    add_idea = sub.add_parser("add-idea", help="Store a research idea")
    add_idea.add_argument("--title", required=True)
    add_idea.add_argument("--observation", required=True)
    add_idea.add_argument("--why", default="")
    add_idea.add_argument("--tags", default="")
    add_idea.add_argument("--context", default="")

    add_web = sub.add_parser("add-web", help="Store a web finding")
    add_web.add_argument("--query", required=True)
    add_web.add_argument("--url", required=True)
    add_web.add_argument("--title", default="")
    add_web.add_argument("--summary", required=True)
    add_web.add_argument("--why", default="")
    add_web.add_argument("--tags", default="")
    add_web.add_argument("--context", default="")

    get_cmd = sub.add_parser("get", help="Get one memory by id")
    get_cmd.add_argument("id", type=int)

    search_cmd = sub.add_parser("search", help="Search memories")
    search_cmd.add_argument("--query", required=True)
    search_cmd.add_argument("--kind", choices=["idea", "web"])
    search_cmd.add_argument("--limit", type=int, default=10)

    recent_cmd = sub.add_parser("recent", help="List recent memories")
    recent_cmd.add_argument("--limit", type=int, default=20)

    upd = sub.add_parser("update", help="Update one memory")
    upd.add_argument("id", type=int)
    upd.add_argument("--title")
    upd.add_argument("--content")
    upd.add_argument("--tags")
    upd.add_argument("--url")
    upd.add_argument("--query")
    upd.add_argument("--context")

    delete_cmd = sub.add_parser("delete", help="Delete one memory")
    delete_cmd.add_argument("id", type=int)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        conn = connect_db()
        ensure_schema(conn)

        if args.cmd == "init":
            print(json.dumps({"ok": True, "db_path": DB_PATH}, indent=2, ensure_ascii=True))
            return

        if args.cmd == "add-idea":
            content = args.observation.strip()
            if args.why.strip():
                content += "\n\nWhy it matters: " + args.why.strip()
            out = add_memory(
                conn,
                kind="idea",
                title=args.title.strip(),
                content=content,
                tags=args.tags,
                context=args.context.strip(),
            )
            print(json.dumps(out, indent=2, ensure_ascii=True))
            return

        if args.cmd == "add-web":
            title = args.title.strip() if args.title.strip() else args.query.strip()
            content = args.summary.strip()
            if args.why.strip():
                content += "\n\nWhy it matters: " + args.why.strip()
            out = add_memory(
                conn,
                kind="web",
                title=title,
                content=content,
                tags=args.tags,
                source_url=args.url.strip(),
                query_text=args.query.strip(),
                context=args.context.strip(),
            )
            print(json.dumps(out, indent=2, ensure_ascii=True))
            return

        if args.cmd == "get":
            print(json.dumps(get_memory(conn, args.id), indent=2, ensure_ascii=True))
            return

        if args.cmd == "search":
            out = search_memories(conn, query=args.query, kind=args.kind, limit=args.limit)
            print(json.dumps(out, indent=2, ensure_ascii=True))
            return

        if args.cmd == "recent":
            out = recent_memories(conn, limit=args.limit)
            print(json.dumps(out, indent=2, ensure_ascii=True))
            return

        if args.cmd == "update":
            if (
                args.title is None
                and args.content is None
                and args.tags is None
                and args.url is None
                and args.query is None
                and args.context is None
            ):
                raise MemoryError("No update fields provided.")
            out = update_memory(
                conn,
                memory_id=args.id,
                title=args.title,
                content=args.content,
                tags=args.tags,
                source_url=args.url,
                query_text=args.query,
                context=args.context,
            )
            print(json.dumps(out, indent=2, ensure_ascii=True))
            return

        if args.cmd == "delete":
            out = delete_memory(conn, args.id)
            print(json.dumps(out, indent=2, ensure_ascii=True))
            return

        raise MemoryError("Unsupported command.")

    except MemoryError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True))
        sys.exit(1)
    except sqlite3.Error as exc:
        print(json.dumps({"error": f"Database error: {exc}"}, ensure_ascii=True))
        sys.exit(1)


if __name__ == "__main__":
    main()
