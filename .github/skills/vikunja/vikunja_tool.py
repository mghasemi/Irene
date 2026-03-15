import argparse
import json
import os
import sys

import requests

URL = os.getenv("VIKUNJA_URL", "http://192.168.1.84:3456")
ALT_URL = os.getenv("VIKUNJA_ALT_URL", "http://mghasemi.ddns.net:3456")
TOKEN = os.getenv("VIKUNJA_TOKEN", "tk_6118b79f059dc5374f4819aa7e4bd3b1ee6ad190")


class VikunjaError(RuntimeError):
    pass


def _parse_response(response):
    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


def _request(method, path, payload=None, params=None, dry_run=False):
    if dry_run:
        return {
            "dry_run": True,
            "method": method,
            "path": path,
            "payload": payload,
            "params": params,
            "attempt_urls": [f"{URL}{path}", f"{ALT_URL}{path}"],
        }

    if not TOKEN:
        raise VikunjaError("Missing Vikunja token. Set VIKUNJA_TOKEN.")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }

    errors = []
    for base_url in (URL, ALT_URL):
        full_url = f"{base_url}{path}"
        try:
            response = requests.request(
                method,
                full_url,
                json=payload,
                params=params,
                headers=headers,
                timeout=10,
            )
        except requests.RequestException as exc:
            errors.append(f"{full_url}: {exc}")
            continue

        if 500 <= response.status_code <= 599:
            snippet = response.text.strip()[:300]
            errors.append(f"{full_url}: HTTP {response.status_code} {snippet}")
            continue

        if response.status_code >= 400:
            body = _parse_response(response)
            raise VikunjaError(
                f"Request failed at {full_url} with HTTP {response.status_code}: "
                f"{json.dumps(body, ensure_ascii=True)}"
            )

        return _parse_response(response)

    raise VikunjaError("Unable to reach Vikunja server. Tried URLs: " + " | ".join(errors))


def list_projects():
    return _request("GET", "/api/v1/projects")


def get_project(project_id):
    return _request("GET", f"/api/v1/projects/{project_id}")


def create_project(title, description=None, dry_run=False):
    payload = {"title": title}
    if description:
        payload["description"] = description
    return _request("PUT", "/api/v1/projects", payload=payload, dry_run=dry_run)


def update_project(project_id, title=None, description=None, dry_run=False):
    payload = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description

    if not payload:
        raise VikunjaError("No fields provided for project update.")

    return _request("POST", f"/api/v1/projects/{project_id}", payload=payload, dry_run=dry_run)


def search_projects(query):
    projects = list_projects()
    if not isinstance(projects, list):
        raise VikunjaError("Unexpected response while listing projects for search.")

    query_norm = query.lower()
    results = []
    for project in projects:
        haystack = " ".join(
            [
                str(project.get("title", "")),
                str(project.get("description", "")),
                str(project.get("identifier", "")),
            ]
        ).lower()
        if query_norm in haystack:
            results.append(project)

    return results


def get_project_by_title(title, exact=False):
    projects = list_projects()
    if not isinstance(projects, list):
        raise VikunjaError("Unexpected response while listing projects for title lookup.")

    title_norm = title.lower()
    if exact:
        matches = [p for p in projects if str(p.get("title", "")).lower() == title_norm]
    else:
        matches = []
        for project in projects:
            project_title = str(project.get("title", "")).lower()
            if title_norm in project_title:
                matches.append(project)

    if not matches:
        raise VikunjaError(f"No project found for title: {title}")
    if len(matches) > 1:
        options = [{"id": p.get("id"), "title": p.get("title")} for p in matches]
        raise VikunjaError(
            "Multiple projects matched. Refine the title or use exact match: "
            + json.dumps(options, ensure_ascii=True)
        )

    return matches[0]


def list_tasks(project_id=None):
    if project_id is not None:
        return _request("GET", f"/api/v1/projects/{project_id}/tasks")
    return _request("GET", "/api/v1/tasks/all")


def get_task(task_id):
    return _request("GET", f"/api/v1/tasks/{task_id}")


def create_task(project_id, title, description=None, due_date=None, dry_run=False):
    payload = {"title": title}
    if description:
        payload["description"] = description
    if due_date:
        payload["due_date"] = due_date
    return _request("PUT", f"/api/v1/projects/{project_id}/tasks", payload=payload, dry_run=dry_run)


def update_task(task_id, title=None, description=None, done=None, due_date=None, project_id=None, dry_run=False):
    payload = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    if done is not None:
        payload["done"] = done
    if due_date is not None:
        payload["due_date"] = due_date
    if project_id is not None:
        payload["project_id"] = project_id

    if not payload:
        raise VikunjaError("No fields provided for task update.")

    return _request("POST", f"/api/v1/tasks/{task_id}", payload=payload, dry_run=dry_run)


def search_tasks(query, project_id=None):
    tasks = list_tasks(project_id)
    if not isinstance(tasks, list):
        raise VikunjaError("Unexpected response while listing tasks for search.")

    query_norm = query.lower()
    results = []
    for task in tasks:
        haystack = " ".join(
            [
                str(task.get("title", "")),
                str(task.get("description", "")),
                str(task.get("identifier", "")),
            ]
        ).lower()
        if query_norm in haystack:
            results.append(task)

    return results


def _parse_bool(value):
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Read/create/update Vikunja projects and tasks with URL failover."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview write requests without sending them to the server.",
    )
    subparsers = parser.add_subparsers(dest="resource", required=True)

    projects = subparsers.add_parser("projects", help="Project operations")
    project_sub = projects.add_subparsers(dest="action", required=True)

    project_sub.add_parser("list", help="List all projects")

    project_search = project_sub.add_parser("search", help="Search projects")
    project_search.add_argument("--query", required=True)

    project_get_by_title = project_sub.add_parser("get-by-title", help="Get one project by title")
    project_get_by_title.add_argument("--title", required=True)
    project_get_by_title.add_argument(
        "--exact",
        action="store_true",
        help="Require exact title match instead of substring match.",
    )

    project_get = project_sub.add_parser("get", help="Get one project")
    project_get.add_argument("project_id", type=int)

    project_create = project_sub.add_parser("create", help="Create a project")
    project_create.add_argument("--title", required=True)
    project_create.add_argument("--description")

    project_update = project_sub.add_parser("update", help="Update a project")
    project_update.add_argument("project_id", type=int)
    project_update.add_argument("--title")
    project_update.add_argument("--description")

    tasks = subparsers.add_parser("tasks", help="Task operations")
    task_sub = tasks.add_subparsers(dest="action", required=True)

    task_list = task_sub.add_parser("list", help="List tasks")
    task_list.add_argument("--project-id", type=int)

    task_search = task_sub.add_parser("search", help="Search tasks")
    task_search.add_argument("--query", required=True)
    task_search.add_argument("--project-id", type=int)

    task_get = task_sub.add_parser("get", help="Get one task")
    task_get.add_argument("task_id", type=int)

    task_create = task_sub.add_parser("create", help="Create a task")
    task_create.add_argument("--project-id", type=int, required=True)
    task_create.add_argument("--title", required=True)
    task_create.add_argument("--description")
    task_create.add_argument("--due-date")

    task_log = task_sub.add_parser("log", help="Log a task (alias of create)")
    task_log.add_argument("--project-id", type=int, required=True)
    task_log.add_argument("--title", required=True)
    task_log.add_argument("--description")
    task_log.add_argument("--due-date")

    task_update = task_sub.add_parser("update", help="Update a task")
    task_update.add_argument("task_id", type=int)
    task_update.add_argument("--title")
    task_update.add_argument("--description")
    task_update.add_argument("--done", type=_parse_bool)
    task_update.add_argument("--due-date")
    task_update.add_argument("--project-id", type=int)

    task_complete = task_sub.add_parser("complete", help="Mark a task as done")
    task_complete.add_argument("task_id", type=int)

    return parser


def _dispatch(args):
    if args.resource == "projects":
        if args.action == "list":
            return list_projects()
        if args.action == "search":
            return search_projects(args.query)
        if args.action == "get-by-title":
            return get_project_by_title(args.title, exact=args.exact)
        if args.action == "get":
            return get_project(args.project_id)
        if args.action == "create":
            return create_project(args.title, args.description, dry_run=args.dry_run)
        if args.action == "update":
            return update_project(args.project_id, args.title, args.description, dry_run=args.dry_run)

    if args.resource == "tasks":
        if args.action == "list":
            return list_tasks(args.project_id)
        if args.action == "search":
            return search_tasks(args.query, args.project_id)
        if args.action == "get":
            return get_task(args.task_id)
        if args.action in {"create", "log"}:
            return create_task(
                args.project_id,
                args.title,
                args.description,
                args.due_date,
                dry_run=args.dry_run,
            )
        if args.action == "update":
            return update_task(
                args.task_id,
                title=args.title,
                description=args.description,
                done=args.done,
                due_date=args.due_date,
                project_id=args.project_id,
                dry_run=args.dry_run,
            )
        if args.action == "complete":
            return update_task(args.task_id, done=True, dry_run=args.dry_run)

    raise VikunjaError("Unsupported command")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = _dispatch(args)
        print(json.dumps(result, indent=2, ensure_ascii=True))
    except VikunjaError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True))
        sys.exit(1)


if __name__ == "__main__":
    main()
