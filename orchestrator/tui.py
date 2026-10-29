"""
Terminal dashboard for the orchestrator: python server.py --tui

uvicorn runs inside the Textual app's event loop, so the dashboard reads the
live Orchestrator / EdgeClient objects directly (no HTTP polling) and shows
everything the server process says: log records, uvicorn's access log, and the
stray print() calls scattered through the handlers.

    Queue       waiting jobs, the job in flight, ok / failed totals
    Downstream  the VL model plus every config.servers entry, re-probed on a timer
    Log         follows the tail; scroll up to read, End to follow again
    Verbose     v also shows this project's DEBUG records: per job the user input, the full system prompt,
                Fuli's memory hits and emotion, the model's token usage / latency / reasoning, and the reply.
                They are collected all along, so switching it on also shows what happened before the keypress.
                A library's own DEBUG chatter never shows. python server.py --tui -v starts with it on.
    Hotkeys     r reload prompt | e reload emotion | d recheck | c clear log | v verbose | q quit

textual is an optional dependency (requirements-tui.txt). server.py imports this
module lazily, so a plain headless run never needs it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import re
import signal
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import uvicorn
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, RichLog, Static

import config

if TYPE_CHECKING:
    from edge_client import EdgeClient
    from orchestrator import Orchestrator

logger = logging.getLogger("reminh.tui")

DOWNSTREAM_POLL_SECONDS = 30  # how often the Downstream panel re-probes (d probes right now)
LOG_MAX_LINES = 5000  # scrollback kept in the log pane
LOG_DRAIN_SECONDS = 0.1  # how often queued log lines are flushed to the pane
LOG_LINES_PER_DRAIN = 300  # per-flush cap, so a log flood can't freeze the UI
GRACEFUL_SHUTDOWN_SECONDS = 5  # on quit, in-flight HTTP requests get this long before uvicorn cancels them
LOG_HISTORY_RECORDS = 5000  # records kept behind the pane: redrawn on a verbose toggle, replayed if startup fails

ReloadFn = Callable[[], Awaitable[dict]]


# ──────────────────────────────────────────────
#  Log pipeline: logging + print() -> queue -> pane
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class LogLine:
    created: float
    level: str  # a logging level name, or "print" / "stderr" for captured console output
    source: str  # logger name; empty for prints
    text: str
    levelno: int = logging.INFO  # below INFO is verbose detail, drawn only while verbose is on


# This project's own code lives next to this file; anything else on the import path is a library.
_APP_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__))) + os.sep
_app_paths: dict[str, bool] = {}


def is_app_record(record: logging.LogRecord) -> bool:
    """True when this project's own code logged the record (judged by source file, not by logger name)."""
    path = record.pathname
    known = _app_paths.get(path)
    if known is None:
        known = _app_paths[path] = os.path.realpath(path).startswith(_APP_DIR)
    return known


class _AppDebugOnly(logging.Filter):
    """INFO and above always pass; DEBUG only from this project's code, never a library's (urllib3, asyncio, ...)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO or is_app_record(record)


class DashboardLogHandler(logging.Handler):
    """
    Hands log records to the dashboard. They arrive on the event loop *and* on
    asyncio.to_thread workers (the Fuli / VL calls), so this only appends to a
    thread-safe queue; the app drains it on a timer.
    """

    def __init__(self, sink: queue.SimpleQueue[LogLine]) -> None:
        super().__init__()
        self.sink = sink
        self.addFilter(_AppDebugOnly())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)  # message plus traceback, when there is one
        except Exception:
            self.handleError(record)
            return
        self.sink.put(LogLine(record.created, record.levelname, record.name, text, record.levelno))


@contextlib.contextmanager
def capture_logging(sink: queue.SimpleQueue[LogLine]) -> Iterator[DashboardLogHandler]:
    """
    Routes the root logger into the dashboard while the block runs, and puts everything back afterwards.

    - server.py's basicConfig() StreamHandler writes straight to the terminal and would scribble over the
      screen, so it is detached.
    - The root level drops to DEBUG so this project's debug records are produced and kept even while
      verbose is off, which is what lets a later keypress show what already happened. The handler's
      filter keeps libraries' own DEBUG chatter out.
    """
    root = logging.getLogger()
    displaced = [h for h in root.handlers if type(h) is logging.StreamHandler]
    previous_level = root.level
    handler = DashboardLogHandler(sink)
    root.addHandler(handler)
    for h in displaced:
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.setLevel(previous_level)
        root.removeHandler(handler)
        for h in displaced:
            root.addHandler(h)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean(text: str) -> str:
    """Third-party output can carry colour codes or carriage-return progress bars; neither belongs in a text UI."""
    return _CONTROL.sub("", _ANSI.sub("", text))


_LEVEL_STYLE = {
    "DEBUG": "dim",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
    "print": "dim",
    "stderr": "yellow",
}
_TEXT_STYLE = {
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
    "print": "dim",
    "stderr": "yellow",
}


def format_log_line(line: LogLine) -> Text:
    # Text, not a markup string: log messages are full of [brackets] that Rich would try to parse
    out = Text()
    stamp = datetime.fromtimestamp(line.created)
    out.append(f"{stamp:%H:%M:%S}.{stamp.microsecond // 1000:03d}", style="dim")
    out.append(f" {line.level:<8}", style=_LEVEL_STYLE.get(line.level, ""))
    if line.source:
        out.append(f" {line.source}", style="cyan")
    style = _TEXT_STYLE.get(line.level, "")
    first, *rest = _clean(line.text).split("\n")
    out.append(f"  {first}", style=style)
    # a multi-line verbose record (a whole prompt, say) gets a gutter so it reads as one block;
    # tracebacks and other multi-line records at INFO and above keep their own layout
    gutter = "    │ " if line.levelno < logging.INFO else ""
    for extra in rest:
        out.append("\n")
        out.append(gutter, style="dim cyan")
        out.append(extra, style=style)
    return out


# ──────────────────────────────────────────────
#  Panel renderers (pure functions, easy to test)
# ──────────────────────────────────────────────
_DOT = "●"


def _fmt_clock(seconds: float) -> str:
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def render_queue(snap: dict) -> Table:
    """snap is Orchestrator.snapshot()."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(no_wrap=True, overflow="ellipsis")

    waiting = snap["queued"]
    grid.add_row("waiting", Text(str(waiting), style="bold yellow" if waiting else "dim"))

    job = snap["in_flight"]
    if job is None:
        grid.add_row("in flight", Text("idle", style="dim"))
    else:
        who = _clean(f"{job['user_name']} ({job['platform']})")
        grid.add_row("in flight", Text(who, style="cyan"))
        grid.add_row("", Text(f"{str(job['request_id'])[:8]} · {job['elapsed_s']:.1f}s", style="dim"))

    done = Text()
    done.append(f"{snap['jobs_ok']} ok", style="green")
    done.append(" · ", style="dim")
    done.append(f"{snap['jobs_failed']} failed", style="red" if snap["jobs_failed"] else "dim")
    grid.add_row("done", done)

    last = snap["last_duration_s"]
    grid.add_row("last job", Text("-", style="dim") if last is None else Text(f"{last:.1f}s"))
    return grid


@dataclass(frozen=True)
class DownstreamResult:
    vl_ok: bool
    vl_reason: str
    servers: dict[str, bool]
    checked_at: float


def _pretty(name: str) -> str:
    """'NV_5090_server_ip' -> 'NV 5090'. Derived, so a new config.servers entry shows up without touching this file."""
    return name.removesuffix("_server_ip").replace("_", " ")


def render_downstream(result: Optional[DownstreamResult]) -> Table:
    """result is None until the first probe has finished."""
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=1)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1)

    def add(ok: Optional[bool], name: str, endpoint: str, up_note: str = "", down_note: str = "unreachable") -> None:
        if not endpoint:  # config.yaml: blank = not deployed yet
            grid.add_row(Text("○", style="dim"), Text(name), Text(""), Text("not set", style="dim"))
        elif ok is None:
            grid.add_row(Text("…", style="dim"), Text(name), Text(endpoint, style="dim"), Text(""))
        elif ok:
            note = Text(up_note, style="dim")
            grid.add_row(Text(_DOT, style="green"), Text(name), Text(endpoint, style="dim"), note)
        else:
            note = Text(_clean(down_note), style="red")
            grid.add_row(Text(_DOT, style="red"), Text(name), Text(endpoint, style="dim"), note)

    vl = config.vl_model
    add(
        None if result is None else result.vl_ok,
        "VL model",
        f"{vl.address}:{vl.port}",
        up_note=vl.alias,
        down_note=(result.vl_reason if result is not None and result.vl_reason else "unreachable"),
    )
    for f in fields(config.servers):
        add(None if result is None else result.servers.get(f.name), _pretty(f.name), getattr(config.servers, f.name))
    return grid


# ──────────────────────────────────────────────
#  The app
# ──────────────────────────────────────────────
class _EmbeddedServer(uvicorn.Server):
    """
    uvicorn.Server without its signal handling. Textual owns the terminal, and
    uvicorn's own SIGINT/SIGTERM capture would swallow the signal, shut the
    server down and then re-raise it on the way out, killing the process before
    Textual has restored the terminal. ServerConsole handles SIGTERM / SIGHUP itself.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class ServerConsole(App[None]):
    TITLE = "Reminh Orchestrator"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #statusbar { height: 1; padding: 0 1; background: $boost; }
    #panels { height: 11; }
    #queue { width: 36; height: 100%; border: round $primary; padding: 0 1; }
    #downstream { width: 1fr; height: 100%; border: round $primary; padding: 0 1; }
    #log { height: 1fr; border: round $primary; padding: 0 1; }
    #log:focus { border: round $accent; }
    """

    BINDINGS = [
        Binding("r", "reload_prompt", "Reload prompt"),
        Binding("e", "reload_emotion", "Reload emotion"),
        Binding("d", "recheck", "Recheck"),
        Binding("c", "clear_log", "Clear"),
        Binding("v", "toggle_verbose", "Verbose"),
        Binding("q", "quit", "Quit"),
        # Textual's own ctrl+c only shows a "press ctrl+q" hint; a headless server stops on ctrl+c, so should this
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        server: uvicorn.Server,
        orchestrator: Orchestrator,
        edge: EdgeClient,
        reload_prompt: ReloadFn,
        reload_emotion: ReloadFn,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._uv = server
        self._orch = orchestrator
        self._edge = edge
        self._reload_fns = {"prompt": reload_prompt, "emotion": reload_emotion}

        # DashboardLogHandler and the captured print() output feed log_sink; _drain_log() moves the records into
        # _history and onto the pane. Records below INFO (the verbose detail) are always kept but only drawn while
        # verbose is on, so switching it on can redraw the pane with what happened before the keypress.
        self.log_sink: queue.SimpleQueue[LogLine] = queue.SimpleQueue()
        self._history: deque[LogLine] = deque(maxlen=LOG_HISTORY_RECORDS)
        self.verbose = verbose
        self._partial: dict[str, str] = {}  # half-written print() lines, per stream

        self._t0 = time.monotonic()
        self._serve_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._down: Optional[DownstreamResult] = None
        self._probing = False

    # ---- state the launcher needs after the screen is gone ----
    @property
    def startup_failed(self) -> bool:
        """True once the server task is over without ever having come up."""
        task = self._serve_task
        return task is not None and task.done() and not self._uv.started

    def tail_lines(self, count: int = 40) -> list[str]:
        """The last `count` log lines as plain text, as far as the current verbose setting shows them."""
        self._collect(None)
        shown = [line for line in self._history if self._shown(line)]
        return [format_log_line(line).plain for line in shown[-count:]]

    # ---- lifecycle ----
    def compose(self) -> ComposeResult:
        yield Static(id="statusbar")
        with Horizontal(id="panels"):
            yield Static(id="queue")
            yield Static(id="downstream")
        # min_width: RichLog renders every line at least this wide, and the default (78) is wider than a 80-column pane
        yield RichLog(id="log", wrap=True, min_width=20, max_lines=LOG_MAX_LINES, auto_scroll=False)
        yield Footer()

    async def on_mount(self) -> None:
        # one row per config.servers entry, one for the VL model, two for the border
        self.query_one("#panels").styles.height = len(fields(config.servers)) + 3
        self.query_one("#queue").border_title = "Queue"
        self.query_one("#downstream").border_title = "Downstream"
        self._refresh_log_title()
        self.query_one("#log").border_subtitle = "PgUp/PgDn scroll · End follows"
        self.query_one("#log").focus()

        self.begin_capture_print(self)  # print() / stderr from anywhere in the process -> on_print()
        self._install_signal_handlers()

        self._serve_task = asyncio.create_task(self._serve(), name="uvicorn")
        self._serve_task.add_done_callback(self._on_server_done)

        self.set_interval(0.5, self._tick)
        self.set_interval(LOG_DRAIN_SECONDS, self._drain_log)
        self.set_interval(DOWNSTREAM_POLL_SECONDS, self._start_probe)
        self._tick()
        self._refresh_downstream()
        self._start_probe(after_startup=True)

    async def _serve(self) -> None:
        try:
            await self._uv.serve()
        except SystemExit as exc:
            # uvicorn sys.exit()s when the lifespan startup fails (edge hub down, ...) or the port is taken.
            # Left alone inside a task, that would tear down the event loop - and the terminal - with it.
            logger.error("server did not start (uvicorn exit code %s); see the log above", exc.code)
        except Exception:
            logger.exception("server crashed")

    def _on_server_done(self, task: asyncio.Task) -> None:
        if self._stopping:
            self.exit()
        elif self.is_running:
            self._tick()  # flips the status bar to "stopped" / "failed to start"

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            # not the main thread, or no signal support on this platform: skip quietly
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, self._on_signal, sig)

    def _on_signal(self, sig: int) -> None:
        logger.info("received %s", signal.Signals(sig).name)
        self.call_later(self.action_quit)

    # ---- periodic refresh ----
    def _tick(self) -> None:
        self._refresh_status()
        self.query_one("#queue", Static).update(render_queue(self._orch.snapshot()))

    def _server_state(self) -> tuple[str, str]:
        task = self._serve_task
        if task is not None and task.done():
            return ("failed to start", "red") if self.startup_failed else ("stopped", "red")
        if self._stopping:
            return "stopping", "yellow"
        if task is None or not self._uv.started:
            return "starting", "yellow"
        return "running", "green"

    def _refresh_status(self) -> None:
        state, state_style = self._server_state()
        # state first: on a narrow terminal the right-hand end is what gets clipped
        bar = Text(no_wrap=True, overflow="ellipsis")
        bar.append("Reminh", style="bold")
        bar.append("  server ")
        bar.append(_DOT, style=state_style)
        bar.append(f" {state}")
        bar.append("  edge ")
        if self._edge.connected:
            bar.append(_DOT, style="green")
            bar.append(" connected")
        elif state == "starting":
            bar.append(_DOT, style="yellow")
            bar.append(" connecting")
        else:
            bar.append(_DOT, style="red")
            bar.append(" disconnected")
        if self.verbose:
            bar.append("  verbose", style="bold magenta")
        bar.append(f"  up {_fmt_clock(time.monotonic() - self._t0)}", style="dim")
        bar.append(f"  {config.server.host}:{config.server.port}", style="dim")
        self.query_one("#statusbar", Static).update(bar)

    def _refresh_downstream(self) -> None:
        panel = self.query_one("#downstream", Static)
        panel.update(render_downstream(self._down))
        if self._probing:
            panel.border_subtitle = "checking..."
        elif self._down is not None:
            panel.border_subtitle = f"checked {datetime.fromtimestamp(self._down.checked_at):%H:%M:%S}"

    def _start_probe(self, after_startup: bool = False) -> None:
        if self._probing:
            return
        self._probing = True  # set here, not in the worker, so two quick calls can't both start one
        self.run_worker(self._probe(after_startup), group="probe", exit_on_error=False)

    async def _probe(self, after_startup: bool = False) -> None:
        self._refresh_downstream()
        try:
            if after_startup:
                # config.servers includes this orchestrator's own port, which only answers once uvicorn is
                # listening; probing earlier would paint it red until the next poll.
                # A failed start ends the wait too, so the panel still fills in.
                while not self._uv.started and not (self._serve_task is not None and self._serve_task.done()):
                    await asyncio.sleep(0.2)
            vl_ok, vl_reason, servers = await self._orch.probe_downstream_servers()
            self._down = DownstreamResult(vl_ok, vl_reason, servers, time.time())
        except Exception as exc:
            logger.warning("downstream probe failed: %s: %s", type(exc).__name__, exc)
        finally:
            self._probing = False
            self._refresh_downstream()

    def _shown(self, line: LogLine) -> bool:
        return self.verbose or line.levelno >= logging.INFO

    def _collect(self, limit: Optional[int]) -> list[LogLine]:
        """Moves queued records into the history and returns them (at most `limit`; None means all of them)."""
        taken: list[LogLine] = []
        while limit is None or len(taken) < limit:
            try:
                taken.append(self.log_sink.get_nowait())
            except queue.Empty:
                break
        self._history.extend(taken)
        return taken

    def _drain_log(self) -> None:
        if self.log_sink.empty():
            return
        pane = self.query_one("#log", RichLog)
        follow = pane.is_vertical_scroll_end  # only chase the tail if the user hasn't scrolled up
        for line in self._collect(LOG_LINES_PER_DRAIN):
            if self._shown(line):
                pane.write(format_log_line(line), scroll_end=follow)

    def _redraw_log(self) -> None:
        """Draws the pane again from the history (after verbose was switched) and ends on the newest line."""
        self._collect(None)
        # newest first, until the pane's own scrollback would be full: a long history costs no more than that
        keep: list[LogLine] = []
        budget = LOG_MAX_LINES
        for line in reversed(self._history):
            if self._shown(line):
                keep.append(line)
                budget -= line.text.count("\n") + 1
                if budget <= 0:
                    break
        pane = self.query_one("#log", RichLog)
        pane.clear()
        for position, line in enumerate(reversed(keep), start=1):
            pane.write(format_log_line(line), scroll_end=position == len(keep))

    def _refresh_log_title(self) -> None:
        self.query_one("#log").border_title = "Log · verbose" if self.verbose else "Log"

    def on_print(self, event: events.Print) -> None:
        """
        Textual redirects sys.stdout / sys.stderr while it runs, which is the only
        reason the print() calls in the handlers (reload messages, Fuli errors, the
        edge client) don't shred the screen. They arrive here in arbitrary chunks.
        """
        stream = "stderr" if event.stderr else "print"
        *lines, self._partial[stream] = (self._partial.get(stream, "") + event.text).split("\n")
        for text in lines:
            if text.strip():
                self.log_sink.put(LogLine(time.time(), stream, "", text))

    # ---- actions ----
    def action_reload_prompt(self) -> None:
        self.run_worker(self._reload("prompt"), group="reload-prompt", exclusive=True, exit_on_error=False)

    def action_reload_emotion(self) -> None:
        self.run_worker(self._reload("emotion"), group="reload-emotion", exclusive=True, exit_on_error=False)

    async def _reload(self, what: str) -> None:
        """Calls the very function behind POST /character/<what>/reload, so key and endpoint can't drift apart."""
        try:
            result = await self._reload_fns[what]()
        except Exception as exc:  # the endpoint functions catch their own errors; this is belt and braces
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if result.get("status") == "success":
            extra = f" (OCEAN {result['OCEAN']})" if "OCEAN" in result else ""
            logger.info("%s config reloaded%s", what, extra)
            self.notify(f"{what} config reloaded", timeout=3)
        else:
            logger.error("%s reload failed: %s", what, result.get("error"))
            self.notify(f"{what} reload failed: {result.get('error')}", severity="error", timeout=8, markup=False)

    def action_recheck(self) -> None:
        if self._probing:
            self.notify("already checking", timeout=2)
        else:
            self._start_probe()

    def action_clear_log(self) -> None:
        # the history too, or switching verbose would bring the cleared lines back
        self._collect(None)
        self._history.clear()
        self.query_one("#log", RichLog).clear()

    def action_toggle_verbose(self) -> None:
        self.verbose = not self.verbose
        state = "on" if self.verbose else "off"
        self._redraw_log()
        self._refresh_log_title()
        self._refresh_status()
        self.notify(f"verbose {state}", timeout=2)

    async def action_quit(self) -> None:
        task = self._serve_task
        if task is None or task.done():
            self.exit()
        elif self._stopping:
            # second press: stop waiting for in-flight requests and the lifespan shutdown
            logger.warning("forcing shutdown")
            self._uv.force_exit = True
            task.cancel()
        else:
            self._stopping = True
            logger.info("shutting down (press q again to force)")
            self._uv.should_exit = True  # uvicorn runs the lifespan shutdown, then _on_server_done() exits the app
            self._tick()


# ──────────────────────────────────────────────
#  Entry point (called from server.py)
# ──────────────────────────────────────────────
def run_with_tui(
    fastapi_app,
    *,
    orchestrator: Orchestrator,
    edge: EdgeClient,
    reload_prompt: ReloadFn,
    reload_emotion: ReloadFn,
    verbose: bool = False,
) -> int:
    """Serves fastapi_app with the dashboard on top. Blocks until the user quits; returns the process exit code."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit("--tui needs an interactive terminal (stdin or stdout is not a TTY)")

    server = _EmbeddedServer(
        uvicorn.Config(
            fastapi_app,
            host=config.server.host,
            port=config.server.port,
            # no uvicorn logging setup: its records propagate to the root logger and land in the log pane
            log_config=None,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        )
    )
    console = ServerConsole(
        server=server,
        orchestrator=orchestrator,
        edge=edge,
        reload_prompt=reload_prompt,
        reload_emotion=reload_emotion,
        verbose=verbose,
    )

    with capture_logging(console.log_sink):
        console.run()

    if console.startup_failed:
        # the alternate screen is gone by now; keep the reason in the scrollback
        print("The server did not start. Last log lines:\n", file=sys.stderr)
        print("\n".join(console.tail_lines()), file=sys.stderr)
        return 1
    return console.return_code or 0
