"""HTTP client for the opencode server, scoped to what the R2C driver needs.

Uses stdlib only (urllib + json) — no `requests` dependency.

Run `python3 scripts/opencode_client.py` for unit self-tests. End-to-end
integration was exercised by scripts/run_hello.py against a live server
(retired 2026-07-04; see git history).
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

try:
    from trusted import trusted_text
except ImportError:  # imported as scripts.opencode_client from the repo root
    from scripts.trusted import trusted_text


DEFAULT_PORT = 4096
# Default timeout for agent-dispatch POSTs. Producer agents (analyzer, coders,
# reviewer) routinely take minutes; the smoke gate stage can take 5–10 min.
# urllib's socket-level timeout aborts the connection if no data is received
# within this window — POST /session/{sid}/message blocks until the assistant
# is done, so this must accommodate the longest expected dispatch.
DEFAULT_TIMEOUT_S = 900

# Server-address resolution (launch-path migration, 2026-06-11). Precedence:
#   1. an explicit URL set via set_server_url() (the driver's --server-url flag)
#   2. the R2C_SERVER_URL environment variable (exported by the r2c opencode
#      plugin into the server process, inherited by backgrounded drivers)
#   3. http://127.0.0.1:<port> from the per-call port argument (legacy path,
#      kept while r2c-start.sh remains the fallback launcher)
SERVER_URL_ENV = "R2C_SERVER_URL"
_SERVER_URL: str | None = None

# This client talks exclusively to the R2C opencode server — never to the
# wider network — so it must never use a proxy. Corporate WSL environments
# set http_proxy/HTTP_PROXY without a loopback exemption, which sent
# 127.0.0.1 API calls to the corporate proxy (a researcher's proxied machine, 2026-06-11:
# Squid answered ERR_CONNECT_FAIL for every request). An empty ProxyHandler
# makes this client immune regardless of the caller's environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def set_server_url(url: str | None) -> None:
    """Pin the server address for every subsequent call (trailing slash
    stripped). Pass None to clear and fall back to env/port resolution."""
    global _SERVER_URL
    _SERVER_URL = url.rstrip("/") if url else None


def _base_url(port: int) -> str:
    import os  # noqa: PLC0415

    if _SERVER_URL:
        return trusted_text(_SERVER_URL)
    env_url = os.environ.get(SERVER_URL_ENV, "").strip()
    if env_url:
        return trusted_text(env_url.rstrip("/"))
    return f"http://127.0.0.1:{int(port)}"


def resolved_server_address(port: int) -> str:
    """The address this client will actually use for `port` — for error
    messages and logs. A driver launched with --server-url or under the
    r2c plugin's R2C_SERVER_URL is NOT talking to 127.0.0.1:<port>, and a
    failure message naming the wrong address costs a debugging round-trip
    (2026-06-11: the first plugin-launched driver failed against a port-0
    serverUrl while its error said 'port 4096')."""
    return _base_url(port)


class OpencodeClientError(Exception):
    pass


class SessionNotFound(OpencodeClientError):
    pass


class ServerUnreachable(OpencodeClientError):
    pass


class DispatchTimeout(OpencodeClientError):
    pass


class DispatchResponseTimeout(DispatchTimeout):
    pass


class DispatchExtensionCeiling(DispatchTimeout):
    """The dispatch POST's total wall-clock crossed the extension ceiling
    while the backend kept generating (failure class:
    unbounded_timeout_extension_on_generating_backend). See
    `dispatch_extension_ceiling_s`."""


# Hard ceiling on the total wall-clock one dispatch POST may occupy.
#
# The declared dispatch timeout is a SOCKET-level read timeout: it fires only
# when the server sends nothing for that long. A backend that keeps
# generating keeps feeding the blocking POST, so every received byte is
# implicit proof of life that re-arms the timer — and the wait extends
# without bound. Live evidence (SRL matrix row, 2026-07-16): a
# paper-fidelity-reviewer dispatch declared timeout=1800s and completed at
# 14019s — one runaway turn burned about four wall-hours the run could not
# get back.
#
# The ceiling bounds that: below it, a slow-but-alive backend keeps its
# extension grace exactly as today; at it, the wait is abandoned with
# DispatchExtensionCeiling — a DispatchTimeout subclass, so the driver
# treats it exactly like the existing backend-alive timeout case (fedavg
# 2026-07-17 path: the error propagates into stage-owned recovery and the
# generating session is abandoned, not killed).
#
# R2C_DISPATCH_EXTENSION_CEILING is a unitless MULTIPLIER of the declared
# dispatch timeout (default 2.0), not an absolute number of seconds —
# declared timeouts vary per agent (600s math-extractor, 1800s fidelity
# reviewer) and the ceiling must scale with each. Values below 1 clamp to 1
# (the ceiling can never undercut the declared timeout, or it would break
# ordinary slow dispatches); unparseable values fall back to the default.
DISPATCH_EXTENSION_CEILING_ENV = "R2C_DISPATCH_EXTENSION_CEILING"
DEFAULT_DISPATCH_EXTENSION_CEILING_MULTIPLIER = 2.0

# Every agent-dispatch budget the driver declares is multiplied by
# R2C_TIMEOUT_SCALE before use. The per-agent budgets in run_pipeline were
# calibrated on a self-hosted Qwen 27B; hosted and free-tier models answer
# slower and get rate-limited, so the default leaves them 3x the room.
# 0 (or any non-positive value) means no practical limit — a hung dispatch
# then never times out. Subprocess budgets for generated code (smoke gate,
# probes) are not dispatches and do not scale.
TIMEOUT_SCALE_ENV = "R2C_TIMEOUT_SCALE"
DEFAULT_TIMEOUT_SCALE = 3.0
_UNLIMITED_TIMEOUT_S = 10 * 365 * 24 * 3600.0


def scaled_timeout_s(timeout_s: float) -> float:
    """The declared dispatch budget times R2C_TIMEOUT_SCALE (default 3.0);
    unparseable values fall back to the default, non-positive means unlimited."""
    raw = os.environ.get(TIMEOUT_SCALE_ENV, "").strip()
    try:
        scale = float(raw) if raw else DEFAULT_TIMEOUT_SCALE
    except ValueError:
        scale = DEFAULT_TIMEOUT_SCALE
    if scale <= 0:
        return _UNLIMITED_TIMEOUT_S
    return timeout_s * scale


def dispatch_extension_ceiling_s(timeout_s: float) -> float:
    """Total wall-clock a message POST may run before the wait is abandoned:
    the ceiling multiplier (env-overridable) times the declared timeout."""
    import os  # noqa: PLC0415

    multiplier = DEFAULT_DISPATCH_EXTENSION_CEILING_MULTIPLIER
    raw = os.environ.get(DISPATCH_EXTENSION_CEILING_ENV, "").strip()
    if raw:
        try:
            multiplier = float(raw)
        except ValueError:
            multiplier = DEFAULT_DISPATCH_EXTENSION_CEILING_MULTIPLIER
    return max(multiplier, 1.0) * timeout_s


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of a single agent dispatch (POST + wait_for_completion)."""

    session_id: str
    user_message_id: str
    assistant_message_id: str
    completed: bool                  # session.idle observed within timeout
    error: dict | None = None        # populated if session.error fired
    elapsed_s: float = 0.0


def _read_with_wall_ceiling(
    resp, *, started: float, ceiling_s: float, timeout: float, base: str,
) -> str:
    """Drain a response body in bounded chunks, enforcing a total wall-clock
    ceiling BETWEEN reads. Each individual read still gets the socket-level
    timeout (a silent backend times out exactly as today), but a backend
    that trickles data forever can no longer extend the wait without bound
    — at the ceiling the wait is abandoned loudly. Worst-case overshoot is
    one socket-timeout window past the ceiling (the read in flight when the
    ceiling passes).

    Reads use `read1` (return whatever bytes are available, blocking only
    until SOME data arrives or EOF) — plain `read(n)` blocks until it fills
    n bytes, which over a trickling stream would never yield control back
    to the wall check. `read1` exists on every http.client.HTTPResponse;
    the plain-read fallback keeps test doubles simple.

    Completeness: unlike plain `read()`, `read1` does not enforce
    Content-Length — a server dying mid-body yields a quiet partial buffer
    (which could even parse as JSON and be accepted as a real response). So
    when the response length is known, an early EOF raises the same loud
    http.client.IncompleteRead the base read path produced."""
    read1 = getattr(resp, "read1", None)
    chunks: list[bytes] = []
    while True:
        elapsed = time.monotonic() - started
        if elapsed > ceiling_s:
            raise DispatchExtensionCeiling(
                f"dispatch POST against {base} crossed the extension "
                f"ceiling: {elapsed:.0f}s elapsed > {ceiling_s:.0f}s ceiling "
                f"(declared timeout {timeout:.0f}s x "
                f"{ceiling_s / timeout if timeout else 0:.1f}); the backend "
                f"was still generating, so the wait is abandoned "
                f"(unbounded_timeout_extension_on_generating_backend) — the "
                f"session keeps generating server-side and is not killed"
            )
        chunk = read1(65536) if read1 is not None else resp.read(65536)
        if not chunk:
            body = b"".join(chunks)
            # http.client.HTTPResponse.length is the REMAINING byte count
            # for Content-Length responses (read1 decrements it); a
            # positive remainder at EOF means the server died mid-body.
            remaining = getattr(resp, "length", None)
            if isinstance(remaining, int) and remaining > 0:
                import http.client  # noqa: PLC0415

                raise http.client.IncompleteRead(body, remaining)
            return body.decode("utf-8", errors="replace")
        chunks.append(chunk)


_WALL_CEILING_UNSET = object()


def _default_wall_ceiling_s(timeout: float) -> float:
    """Ceiling for calls that declare no explicit one: 2x the socket
    timeout, floored at timeout+120s so short-timeout calls keep a usable
    grace window. Every _http call is wall-bounded — the 2026-08-23 A8
    driver sat ~19h past every declared deadline on one established
    socket (exact site unprovable post-mortem; the process was killed
    before a stack sample), and per-read socket timeouts provably cannot
    bound a trickling response on their own."""
    return max(2.0 * timeout, timeout + 120.0)


def _http(
    path: str,
    *,
    port: int,
    method: str = "GET",
    data: dict | None = None,
    timeout: float = 30.0,
    wall_ceiling_s: float | object = _WALL_CEILING_UNSET,
) -> tuple[int, str]:
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    base = _base_url(port)
    req = urllib.request.Request(
        f"{base}{trusted_text(path)}", method=method, data=body, headers=headers
    )
    if wall_ceiling_s is _WALL_CEILING_UNSET:
        wall_ceiling_s = _default_wall_ceiling_s(timeout)
    # Monotonic: the ceiling measures multi-hour wall windows, exactly the
    # domain where an NTP step on time.time() would corrupt the arithmetic.
    started = time.monotonic()
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            if wall_ceiling_s is None:
                return r.status, r.read().decode("utf-8", errors="replace")
            return r.status, _read_with_wall_ceiling(
                r, started=started, ceiling_s=wall_ceiling_s,
                timeout=timeout, base=base)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", None)
        timed_out = (
            isinstance(e, (TimeoutError, socket.timeout))
            or isinstance(reason, (TimeoutError, socket.timeout))
            or "timed out" in str(e).lower()
        )
        message_post = (
            method.upper() == "POST"
            and path.startswith("/session/")
            and path.endswith("/message")
        )
        if timed_out and message_post:
            raise DispatchResponseTimeout(
                f"dispatch POST timed out after {timeout}s against {base}; "
                "the opencode server may still be processing the agent message"
            ) from e
        raise ServerUnreachable(f"opencode server unreachable at {base}: {e}") from e


def health_check(*, port: int = DEFAULT_PORT) -> bool:
    """True iff the server responds to GET /session within a short timeout."""
    try:
        status, _ = _http("/session", port=port, timeout=3.0)
        return status == 200
    except ServerUnreachable:
        return False


def list_sessions(*, port: int = DEFAULT_PORT) -> list[dict]:
    status, body = _http("/session", port=port)
    if status != 200:
        raise OpencodeClientError(f"GET /session returned {status}: {body[:200]}")
    return json.loads(body)


def list_agents(*, port: int = DEFAULT_PORT) -> list[dict]:
    """GET /agent — each entry's `model` field is the agent's resolved model
    (frontmatter + extends-chain merged by the server). For agents with no
    declared model (e.g. `build`), `model` is absent."""
    status, body = _http("/agent", port=port)
    if status != 200:
        raise OpencodeClientError(f"GET /agent returned {status}: {body[:200]}")
    return json.loads(body)


def get_agent_model(name: str, *, port: int = DEFAULT_PORT) -> dict | None:
    """Return the resolved {providerID, modelID} for an agent, or None if
    the agent has no declared model (falls back to session default)."""
    for a in list_agents(port=port):
        if a.get("name") == name:
            return a.get("model")
    raise OpencodeClientError(f"agent {name!r} not found on server")


def discover_session(*, directory: str, port: int = DEFAULT_PORT) -> str:
    """Find the active session ID for the given directory.

    Strategy: filter all sessions by `directory`, sort by `time.updated`
    descending, return the most recent. The user's open TUI is the most
    recently touched session whose `directory` matches the workspace.
    """
    matches = [s for s in list_sessions(port=port) if s.get("directory") == directory]
    if not matches:
        raise SessionNotFound(
            f"no session found with directory={directory!r} (have {len(list_sessions(port=port))} total)"
        )
    matches.sort(key=lambda s: s.get("time", {}).get("updated", 0), reverse=True)
    return matches[0]["id"]


def post_message(
    *,
    session_id: str,
    agent: str,
    prompt: str,
    model: dict | None = None,
    port: int = DEFAULT_PORT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """POST a user message and BLOCK until the assistant finishes responding.

    The server holds the response open until the LLM is done — so `timeout_s`
    must accommodate the longest expected agent dispatch. Returns the parsed
    response (assistant message info dict).

    `model` is `{"providerID": ..., "modelID": ...}`. If None, the server uses
    the session's currently-selected default (set by the TUI / opencode.json).
    """
    body: dict = {
        "parts": [{"type": "text", "text": prompt}],
        "agent": agent,
    }
    if model is not None:
        body["model"] = model
    status, resp = _http(
        f"/session/{session_id}/message", port=port, method="POST",
        data=body, timeout=timeout_s,
        # `timeout_s` is a per-read socket timeout, so a still-generating
        # backend extends the wait indefinitely (proof of life by trickle);
        # the wall ceiling bounds the total extension. See
        # dispatch_extension_ceiling_s.
        wall_ceiling_s=dispatch_extension_ceiling_s(timeout_s))
    if status != 200:
        raise OpencodeClientError(f"POST /session/{session_id}/message returned {status}: {resp[:300]}")
    return json.loads(resp) if resp else {}


def abort_session(session_id: str, *, port: int = DEFAULT_PORT) -> bool:
    """Best-effort abort of a session's in-flight generation.

    An abandoned turn is not a passive leak: it keeps GENERATING and
    WRITING server-side after the driver walks away. The 2026-08-24
    test-generator wander proved it — the scope guard quarantined the
    turn's stray docs/ writes, and the still-running turn re-created
    them and edited a second tracked file MINUTES AFTER the halt. Every
    path that abandons a live turn must abort it first. Best-effort by
    design: a failed abort must never mask the original failure."""
    try:
        status, _ = _http(
            f"/session/{session_id}/abort", port=port, method="POST",
            data={}, timeout=10.0,
        )
        return status == 200
    except OpencodeClientError:
        return False


def create_session(
    *,
    title: str,
    parent_id: str | None = None,
    directory: str | None = None,
    port: int = DEFAULT_PORT,
) -> str:
    """Create a new opencode session and return its id.

    Used by the per-dispatch-session transport: each agent dispatch runs in a
    FRESH session so a prior stage's conversation cannot bleed into it. The
    shared-session model let the stage_3a notebook-generator drift into
    rewriting a stale stage_2c review (GBALD, 2026-06-29) — isolation makes
    the prompts' "ignore the transcript" instruction structural instead of
    advisory. The pattern was first proven in run_explainer_experiment.py
    (retired 2026-07-04, in git history);
    this is the shared, production version.

    `parent_id` nests the session under an existing one (the user's TUI
    session) so the TUI renders work sessions as its children and they are
    disposed with the parent. `directory` pins the workspace so the agent's
    file tools and agent/config resolution match the parent. Both are
    optional so the helper also covers the flat-session case.

    The server takes `directory` as a query param and `{title, parentID}` in
    the body (opencode SDK SessionCreateData).
    """
    body: dict = {}
    if title:
        body["title"] = title
    if parent_id:
        body["parentID"] = parent_id
    path = "/session"
    if directory:
        path = f"/session?directory={urllib.parse.quote(directory, safe='')}"
    status, resp = _http(path, port=port, method="POST", data=body)
    if status != 200:
        raise OpencodeClientError(f"POST {path} returned {status}: {resp[:300]}")
    sid = (json.loads(resp).get("id") if resp else None)
    if not sid:
        raise OpencodeClientError(
            f"POST {path} returned 200 but no session id in body: {resp[:300]}"
        )
    return sid


# The server's per-step output-token budget. A turn whose FINAL step burned
# exactly this many output tokens without issuing a tool call is the
# deterministic cap-burn signature (overnight 2026-07-08, ACC run 5: three
# casualty turns, all measured at exactly this value with no tool part).
# Must equal the served model's max output tokens for the detector to be
# right in either direction. The driver resolves it from the server at
# startup (configure_output_cap); R2C_PER_STEP_OUTPUT_CAP pins it instead;
# 32768 is the fallback when neither is available.
PER_STEP_OUTPUT_CAP = int(os.environ.get("R2C_PER_STEP_OUTPUT_CAP", "32768"))


def default_model_output_limit(*, port: int = DEFAULT_PORT) -> int | None:
    """limit.output of the server's default model (GET /config `model`,
    looked up in GET /config/providers), or None when unavailable."""
    status, body = _http("/config", port=port)
    if status != 200:
        return None
    model = (json.loads(body).get("model") or "")
    provider_id, _, model_id = model.partition("/")
    if not provider_id or not model_id:
        return None
    status, body = _http("/config/providers", port=port)
    if status != 200:
        return None
    for provider in json.loads(body).get("providers") or []:
        if provider.get("id") != provider_id:
            continue
        limit = ((provider.get("models") or {}).get(model_id) or {}).get("limit") or {}
        out = limit.get("output")
        return out if isinstance(out, int) and out > 0 else None
    return None


def configure_output_cap(*, port: int = DEFAULT_PORT) -> int:
    """Point PER_STEP_OUTPUT_CAP at the served model's output limit unless
    R2C_PER_STEP_OUTPUT_CAP pins it. Best-effort: any failure keeps the
    current value, so a startup probe can never block a run."""
    global PER_STEP_OUTPUT_CAP
    if os.environ.get("R2C_PER_STEP_OUTPUT_CAP"):
        return PER_STEP_OUTPUT_CAP
    try:
        out = default_model_output_limit(port=port)
    except Exception:  # noqa: BLE001 — best-effort probe
        out = None
    if out:
        PER_STEP_OUTPUT_CAP = out
    return PER_STEP_OUTPUT_CAP


def fetch_session_messages(
    session_id: str, *, port: int = DEFAULT_PORT
) -> list[dict]:
    """The session's full message list (info + parts), for post-dispatch
    turn-shape classification. Raises OpencodeClientError / ServerUnreachable
    on any failure — callers classifying best-effort catch and degrade."""
    status, body = _http(f"/session/{session_id}/message", port=port)
    if status != 200:
        raise OpencodeClientError(
            f"GET /session/{session_id}/message returned {status}: {body[:200]}")
    try:
        messages = json.loads(body)
    except json.JSONDecodeError as e:
        raise OpencodeClientError(
            f"GET /session/{session_id}/message returned non-JSON: {e}")
    return messages if isinstance(messages, list) else []


def _split_steps(parts: list) -> list[list[dict]]:
    """Group a message's parts into steps on step-start markers."""
    steps: list[list[dict]] = []
    current: list[dict] | None = None
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "step-start":
            current = []
            steps.append(current)
        else:
            if current is None:  # parts before any step-start marker
                current = []
                steps.append(current)
            current.append(part)
    return steps


def cap_burn_steps(
    messages: list[dict], *, cap: int | None = None
) -> list[dict]:
    """Every cap-burned step across a session's assistant messages, oldest
    first (item 23's cap-burn event, the item 8 taxonomy datum).

    A step is a burn when its step-finish records output tokens at the
    per-step cap and no VALID tool call landed in it. Two measured shapes
    (both live on 2026-07-13):
      - reasoning burn: no tool part at all — the model spent the whole
        step budget composing and the write was never issued (ACC
        2026-07-08, the original signature).
      - truncated tool call: the write itself outgrew the cap and its
        JSON arrived cut off — the server records it as a tool part whose
        tool name is the literal "invalid" (ICRA / iDb-RRT / ms3d
        2026-07-13, the shape the final-step-only detector missed).
    The scan covers every step of every assistant message: ICRA's burn
    sat one message before a short stub step and was mislabeled
    short_empty by the final-step-only version.

    Deliberately conservative: a step with no step-finish part (some
    errored turns never get one) cannot be measured and is skipped — the
    caller reports unmeasured rather than guessed. Whether a burn was
    fatal or self-recovered is the CALLER's context: a dispatch that
    wrote nothing died with it, a dispatch that succeeded recovered."""
    if cap is None:
        cap = PER_STEP_OUTPUT_CAP
    burns: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        for step in _split_steps(message.get("parts") or []):
            tools = [p for p in step if p.get("type") == "tool"]
            if any(p.get("tool") != "invalid" for p in tools):
                continue  # a valid tool call landed: not a burn
            finish = next(
                (p for p in step if p.get("type") == "step-finish"), None)
            if finish is None:
                continue
            output = (finish.get("tokens") or {}).get("output")
            if not isinstance(output, (int, float)) or isinstance(output, bool):
                continue
            if output < cap:
                continue
            burns.append({
                "output_tokens": int(output),
                "cap": cap,
                "tool_call_truncated": bool(tools),
            })
    return burns


def completed_tool_calls(messages: list[dict]) -> int:
    """How many valid tool calls a session actually completed (R2C-060).

    The signal that separates two shapes of dead turn that look identical
    from the outside. A turn that read seven files and then died composing
    its write has all that work sitting in its session context, so resuming
    is nearly free; a turn that died having done nothing has nothing to
    preserve and is better re-rolled with a clean context.

    Counts valid tool parts only: the literal `invalid` tool name is the
    server's record of a tool call whose JSON arrived truncated, which is a
    death rather than a completed step."""
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        for part in message.get("parts") or []:
            if part.get("type") == "tool" and part.get("tool") != "invalid":
                total += 1
    return total


def final_step_output_tokens(messages: list[dict]) -> int | None:
    """Output tokens of the last measurable step, or None when unmeasurable.

    A one-token final step is an inference-side immediate stop, transient by
    nature, and an in-session retry costs one prompt (R2C-060)."""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        for step in reversed(_split_steps(message.get("parts") or [])):
            finish = next(
                (p for p in step if p.get("type") == "step-finish"), None)
            if finish is None:
                continue
            output = (finish.get("tokens") or {}).get("output")
            if isinstance(output, (int, float)) and not isinstance(output, bool):
                return int(output)
    return None


def session_activity_fingerprint(
    session_id: str, *, port: int = DEFAULT_PORT
) -> str | None:
    """Reduce a session's message list to a comparable activity fingerprint.

    Two equal fingerprints a settle-pause apart mean the session is
    quiescent — the rung-3 liveness precheck (a re-dispatch over a session
    still generating would be two writers on one run dir; the T1b hazard).
    Returns None when the session does not exist (a gone session cannot be
    writing). Raises OpencodeClientError/ServerUnreachable when the server
    cannot answer — the caller treats that as NOT quiescent (conservative)."""
    status, body = _http(f"/session/{session_id}/message", port=port)
    if status == 404:
        return None
    if status != 200:
        raise OpencodeClientError(
            f"GET /session/{session_id}/message returned {status}: {body[:200]}")
    try:
        messages = json.loads(body)
    except json.JSONDecodeError as e:
        raise OpencodeClientError(
            f"GET /session/{session_id}/message returned non-JSON: {e}")
    if not isinstance(messages, list):
        messages = []
    last = messages[-1] if messages else {}
    info = last.get("info") or last if isinstance(last, dict) else {}
    last_id = str(info.get("id") or "")
    completed = str(((info.get("time") or {}).get("completed"))
                    if isinstance(info.get("time"), dict) else "")
    parts = last.get("parts") if isinstance(last, dict) else None
    n_parts = len(parts) if isinstance(parts, list) else 0
    return f"{len(messages)}:{last_id}:{completed}:{n_parts}"


def dispatch_and_wait(
    *,
    session_id: str,
    agent: str,
    prompt: str,
    model: dict | None = None,
    port: int = DEFAULT_PORT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> DispatchResult:
    """POST a message and wait for the assistant to finish.

    Thin wrapper around `post_message`. The server holds the POST open until
    the LLM is done, so the POST itself is the wait — no SSE listener needed.
    Returns a DispatchResult with the assistant message ID and elapsed time.
    Raises DispatchResponseTimeout on a blocking POST read timeout. In that
    case urllib aborts the client connection even though the server may
    continue processing and write the requested artifact.
    """
    started = time.time()
    response = post_message(
        session_id=session_id, agent=agent, prompt=prompt, model=model,
        port=port, timeout_s=timeout_s,
    )
    info = response.get("info") or {}
    error = info.get("error")
    if error:
        raise OpencodeClientError(
            f"dispatch produced an assistant error: {json.dumps(error)[:300]}"
        )
    return DispatchResult(
        session_id=session_id,
        user_message_id=info.get("parentID", ""),
        assistant_message_id=info.get("id", ""),
        completed=True,
        elapsed_s=time.time() - started,
    )


def _self_test() -> None:
    # Pure unit tests — no live server required.

    # discover_session logic via monkey-patching list_sessions
    import sys
    sample_sessions = [
        {"id": "ses_A", "directory": "/other", "time": {"updated": 100}},
        {"id": "ses_B", "directory": "/workspace", "time": {"updated": 200}},
        {"id": "ses_C", "directory": "/workspace", "time": {"updated": 300}},
    ]
    saved = sys.modules[__name__].list_sessions
    try:
        sys.modules[__name__].list_sessions = lambda *, port=4096: sample_sessions  # type: ignore
        assert discover_session(directory="/workspace") == "ses_C"  # most recent
        try:
            discover_session(directory="/nowhere")
            raise AssertionError("expected SessionNotFound")
        except SessionNotFound:
            pass
    finally:
        sys.modules[__name__].list_sessions = saved

    # DispatchResult is a clean dataclass
    r = DispatchResult(session_id="ses_x", user_message_id="m1", assistant_message_id="m2", completed=True, elapsed_s=1.5)
    assert r.session_id == "ses_x"
    assert r.completed is True

    # create_session: title + parentID go in the body, directory in the query
    # string, and the returned id is read from the response body.
    saved_http = sys.modules[__name__]._http
    captured: dict = {}

    def _fake_http(path, *, port, method="GET", data=None, timeout=30.0):
        captured["path"] = path
        captured["method"] = method
        captured["data"] = data
        return 200, json.dumps({"id": "ses_new", "parentID": (data or {}).get("parentID")})

    try:
        sys.modules[__name__]._http = _fake_http  # type: ignore[assignment]
        sid = create_session(
            title="r2c x", parent_id="ses_parent", directory="/ws space"
        )
        assert sid == "ses_new", sid
        assert captured["method"] == "POST"
        assert captured["path"].startswith("/session?directory="), captured["path"]
        # directory is percent-encoded (space → %20), parentID rides the body.
        assert "%20" in captured["path"], captured["path"]
        assert captured["data"]["parentID"] == "ses_parent"
        assert captured["data"]["title"] == "r2c x"
        # No directory → bare /session path, no query string.
        captured.clear()
        create_session(title="flat")
        assert captured["path"] == "/session", captured["path"]
        assert "parentID" not in captured["data"]
        # 200 with no id is an error, not a silent empty session.
        sys.modules[__name__]._http = (  # type: ignore[assignment]
            lambda path, *, port, method="GET", data=None, timeout=30.0: (200, "{}")
        )
        try:
            create_session(title="bad")
            raise AssertionError("expected OpencodeClientError for id-less 200")
        except OpencodeClientError:
            pass
    finally:
        sys.modules[__name__]._http = saved_http  # type: ignore[assignment]

    # A blocking message POST timeout is not the same as connection refusal:
    # opencode may keep processing the agent turn and still write artifacts.
    saved_open = _OPENER.open
    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    try:
        _OPENER.open = _raise_timeout  # type: ignore[assignment]
        try:
            post_message(session_id="ses_x", agent="build", prompt="hello", timeout_s=1)
            raise AssertionError("expected DispatchResponseTimeout")
        except DispatchResponseTimeout:
            pass
    finally:
        _OPENER.open = saved_open  # type: ignore[assignment]

    # Server-URL resolution precedence: explicit > env > port.
    import os as _os
    assert _base_url(4096) == "http://127.0.0.1:4096"
    _os.environ[SERVER_URL_ENV] = "http://127.0.0.1:39021/"
    try:
        assert _base_url(4096) == "http://127.0.0.1:39021"
        set_server_url("http://127.0.0.1:40000/")
        assert _base_url(4096) == "http://127.0.0.1:40000"
    finally:
        set_server_url(None)
        del _os.environ[SERVER_URL_ENV]
    assert _base_url(4097) == "http://127.0.0.1:4097"

    # Proxy immunity, tested behaviorally (the proxied-researcher scenario): with
    # http_proxy pointing at a dead address and no loopback exemption, a
    # request to a local server must still succeed. The default opener
    # would route it to the dead proxy and hang; _OPENER must not.
    import http.server as _hs
    import threading as _threading

    class _Quiet(_hs.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    server = _hs.HTTPServer(("127.0.0.1", 0), _Quiet)
    _threading.Thread(target=server.serve_forever, daemon=True).start()
    poisoned = {"http_proxy": "http://192.0.2.1:9", "no_proxy": ""}
    saved_env = {k: _os.environ.get(k) for k in poisoned}
    try:
        _os.environ.update(poisoned)
        with _OPENER.open(
            f"http://127.0.0.1:{server.server_address[1]}/", timeout=5
        ) as r:
            assert r.read() == b"ok"
    finally:
        for k, v in saved_env.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        server.shutdown()

    # health_check on a definitely-unused port returns False (no exception)
    assert health_check(port=58923) is False

    # discover_session against an unreachable server raises ServerUnreachable
    try:
        list_sessions(port=58923)
        raise AssertionError("expected ServerUnreachable")
    except ServerUnreachable:
        pass

    # list_agents on an unreachable server also raises ServerUnreachable
    try:
        list_agents(port=58923)
        raise AssertionError("expected ServerUnreachable")
    except ServerUnreachable:
        pass

    print("opencode_client.py: unit self-tests passed")


if __name__ == "__main__":
    _self_test()
