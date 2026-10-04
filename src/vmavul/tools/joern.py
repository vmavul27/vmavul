"""Joern 4.0.547 characterization of D2 (taint length), D3 (control-flow form) and D4
(source-referencing guard), following the rules of Section 3.2:

* **D2** -- sinks are the vulnerability-relevant arguments of the family's sink calls /
  operators; candidate sources are function parameters, values returned by external
  input calls, global variables and locally read values (buffers filled by read-like
  calls); for Use After Free the deallocated pointer is the source.  Joern recovers the
  intra-procedural flows; the taint length is the number of **def-use edges** on the
  shortest recovered path, i.e. the number of transitions between distinct statements
  along the path (a parameter counts as defined at function entry; a source evaluated
  inside the sink statement has length 0).
* **D3** -- the control structures on which the sink is **transitively control
  dependent** (Joern ``controlledBy``, i.e. the CDG), counted per control statement (a
  compound condition is one structure).  A sink inside a ``catch`` handler additionally
  depends on that handler (Joern's CFG has no exceptional edges, so this is taken from
  the enclosing ``CATCH``; a ``try`` body does not count).
* **D4** -- the predicates of those structures; Guarded iff at least one predicate
  references the D2 source or a value data-dependent on it (Joern ``reachableBy``).

One ``importCode`` + one query per analysed unit; the result is written to a JSON file.
"""

from __future__ import annotations

import json
import re
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class JoernError(RuntimeError):
    pass


@dataclass
class JoernAnalysis:
    target_found: bool
    sinks: list[dict] = field(default_factory=list)       # candidate sinks (code, line)
    sources: list[dict] = field(default_factory=list)     # candidate sources (code, line)
    taint_edges: Optional[int] = None                     # None = no source->sink flow
    flow: list[dict] = field(default_factory=list)        # statements on the shortest path
    sink: Optional[dict] = None                           # primary sink
    source: Optional[dict] = None                         # source of the shortest path
    structures: list[dict] = field(default_factory=list)  # {kind, line, code}
    predicates: list[dict] = field(default_factory=list)  # {code, line, refs_source}

    @property
    def guarded(self) -> bool:
        return any(p.get("refs_source") for p in self.predicates)


def _s(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _str_set(items) -> str:
    return "Set[String](" + ", ".join(_s(i) for i in items) + ")"


def _idx_map(m: dict) -> str:
    parts = [f"{_s(k)} -> List[Int]({', '.join(str(int(i)) for i in (v or []))})" for k, v in (m or {}).items()]
    return "Map[String, List[Int]](" + ", ".join(parts) + ")"


_BODY = r'''
  def esc(s: String): String =
    s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "").replace("\t", " ")
      .filter(ch => ch >= ' ')
  def ln(n: StoredNode): Int = n match {
    case c: CfgNode => c.lineNumber.map(_.toInt).getOrElse(-1)
    case _ => -1
  }
  def codeOf(n: StoredNode): String = n match {
    case c: CfgNode => c.code
    case _ => ""
  }
  def parentOf(n: StoredNode): Option[StoredNode] = n._astIn.nextOption()
  def isBoundary(n: StoredNode): Boolean =
    n.isInstanceOf[Block] || n.isInstanceOf[Method] || n.isInstanceOf[ControlStructure]
  def stmtOf(n: StoredNode): Long = n match {
    case p: MethodParameterIn => p.id
    case _ =>
      var cur: StoredNode = n
      var par = parentOf(cur)
      while (par.isDefined && !isBoundary(par.get)) { cur = par.get; par = parentOf(cur) }
      cur.id
  }
  def kindOf(t: String): String = t match {
    case "IF" | "SWITCH" | "ELSE" => "conditional"
    case "FOR" | "WHILE" | "DO" => "loop"
    case "CATCH" => "exception"
    case _ => ""
  }
  def jsonNode(n: StoredNode): String = "{\"code\":\"" + esc(codeOf(n).take(200)) + "\",\"line\":" + ln(n) + "}"
  def arr(xs: List[String]): String = xs.mkString("[", ",", "]")

  val cands = mycpg.method.nameExact(tname).filter(_.isExternal == false).l
  val payload = if (cands.isEmpty) "{\"target_found\":false}" else {
    val m = cands.head
    val calls = m.call.l
    def argsOf(c: Call, idx: List[Int]): List[Expression] =
      if (idx.isEmpty) c.argument.l else c.argument.l.filter(a => idx.contains(a.argumentIndex))
    // ---------------- sources
    val paramNames = m.parameter.name.toSet
    val localNames = m.local.name.toSet
    val globalNames = mycpg.method.nameExact("<global>").local.name.toSet -- localNames -- paramNames
    val globalIds: List[CfgNode] = m.ast.isIdentifier.filter(i => globalNames.contains(i.name)).l
    val sources: List[CfgNode] = mode match {
      case "uaf" => calls.filter(c => deallocs.contains(c.name)).flatMap(c => c.argument.l.filter(_.argumentIndex == 1))
      case "leak" => calls.filter(c => acquires.contains(c.name))
      case _ =>
        m.parameter.l ++ globalIds ++
          // values returned by external calls (callee not defined in the unit) + configured input calls
          calls.filter(c => retSrc.contains(c.name) ||
            (!c.name.startsWith("<operator>") && c.callee.l.forall(_.isExternal))) ++
          calls.filter(c => outSrc.contains(c.name)).flatMap(c => argsOf(c, outSrc(c.name)))
    }
    val sourceIds = sources.map(_.id).toSet
    // ---------------- sinks: (sink call, relevant argument)
    val sinkPairs: List[(CfgNode, CfgNode)] = (if (mode == "leak") {
      calls.filter(c => !acquires.contains(c.name) && !releases.contains(c.name) && !c.name.startsWith("<operator>.assignment"))
        .flatMap(c => c.argument.l.map(a => (c: CfgNode, a: CfgNode))) ++
        m.ast.isReturn.l.flatMap(r => Iterator.single(r).astChildren.l.collect { case e: Expression => (r: CfgNode, e: CfgNode) })
    } else {
      calls.filter(c => sinkCallArgs.contains(c.name)).flatMap(c => argsOf(c, sinkCallArgs(c.name)).map(a => (c: CfgNode, a: CfgNode))) ++
        calls.filter(c => sinkOpArgs.contains(c.name)).flatMap(c => argsOf(c, sinkOpArgs(c.name)).map(a => (c: CfgNode, a: CfgNode)))
    })  // a sink operand that is itself a source (e.g. `100 / atoi(s)`) yields a 0-edge flow
    // ---------------- flows and def-use edge counts
    // Def-use edges: transitions between distinct statements on the path.  Predicates of
    // control structures only USE a value (no definition), so an intermediate predicate on
    // the path is not counted as a propagation step.
    val condIds: Set[Long] = m.controlStructure.condition.id.toSet
    def edges(f: io.joern.dataflowengineoss.language.Path): Int = {
      val stmts = f.elements.map(e => stmtOf(e))
      val compressed = stmts.foldLeft(List.empty[Long])((acc, s) => if (acc.headOption.contains(s)) acc else s :: acc).reverse
      val kept = compressed.zipWithIndex.filter { case (sid, i) =>
        i == 0 || i == compressed.size - 1 || !condIds.contains(sid) }
      kept.size - 1
    }
    val flows = sinkPairs.flatMap { case (sinkCall, arg) =>
      Iterator.single(arg).reachableByFlows(sources.iterator).l.filter(f => f.elements.nonEmpty)
        .map(f => (sinkCall, f, edges(f)))
    }.filter { case (sc, f, e) => !(mode == "uaf" && e == 0) }
    val best = if (flows.isEmpty) None else Some(flows.minBy { case (sc, f, e) => (e, ln(sc)) })
    val primary: Option[CfgNode] = best.map(_._1).orElse(sinkPairs.headOption.map(_._1))
    val srcNode: Option[CfgNode] = best.flatMap(_._2.elements.headOption.collect { case c: CfgNode => c })
    // ---------------- D3 structures (transitive control dependence)
    case class Struct(kind: String, id: Long, line: Int, code: String, pred: Option[StoredNode])
    def structureOf(c: StoredNode): Option[Struct] = {
      var prev: StoredNode = c
      var cur = parentOf(c)
      var result: Option[Struct] = None
      var done = false
      while (!done && cur.isDefined) {
        cur.get match {
          case cs: ControlStructure =>
            val k = kindOf(cs.controlStructureType)
            if (cs.condition.id.l.contains(prev.id)) result = if (k.isEmpty) None else Some(Struct(k, cs.id, ln(cs), cs.code.takeWhile(_ != '{'), Some(prev)))
            else result = Some(Struct("conditional", prev.id, ln(prev), codeOf(prev), Some(prev)))
            done = true
          case _: Block | _: Method =>
            result = Some(Struct("conditional", prev.id, ln(prev), codeOf(prev), Some(prev)))
            done = true
          case _ =>
            prev = cur.get
            cur = parentOf(prev)
        }
      }
      result
    }
    val structs: List[Struct] = primary.map { s =>
      val cdg = Iterator.single(s).controlledBy.l.flatMap(c => structureOf(c))
      var anc = List.empty[Struct]
      var prev: StoredNode = s
      var cur = parentOf(s)
      while (cur.isDefined) {
        cur.get match {
          case cs: ControlStructure if cs.controlStructureType == "CATCH" =>
            anc = Struct("exception", cs.id, ln(cs), "catch", None) :: anc
          case _ => ()
        }
        prev = cur.get
        cur = parentOf(prev)
      }
      (cdg ++ anc).groupBy(_.id).values.map(_.head).toList.sortBy(_.line)
    }.getOrElse(List.empty)
    // ---------------- D4 predicates referencing the source
    val srcList: List[CfgNode] = srcNode.map(List(_)).getOrElse(sources)
    val srcNames: Set[String] = srcList.flatMap {
      case p: MethodParameterIn => List(p.name)
      case i: Identifier => List(i.name)
      case c: Call =>
        parentOf(c).toList.collect { case a: Call if a.name == "<operator>.assignment" => a.argument.l.filter(_.argumentIndex == 1).map(_.code) }.flatten
      case _ => List.empty[String]
    }.toSet
    def refsSource(p: StoredNode): Boolean = p match {
      case e: Expression =>
        val ids = Iterator.single(e).ast.isIdentifier.l.distinctBy(_.id)
        ids.exists(i => srcNames.contains(i.name) || Iterator.single(i).reachableBy(srcList.iterator).nonEmpty)
      case _ => false
    }
    val preds = structs.filter(_.pred.isDefined).map(st => (st, refsSource(st.pred.get)))
    def flowJson(f: io.joern.dataflowengineoss.language.Path): String =
      arr(f.elements.map(e => jsonNode(e)))
    "{\"target_found\":true" +
      ",\"sinks\":" + arr(sinkPairs.map(_._1).distinctBy(_.id).take(40).map(jsonNode)) +
      ",\"sources\":" + arr(sources.take(40).map(jsonNode)) +
      ",\"taint_edges\":" + best.map(_._3.toString).getOrElse("null") +
      ",\"flow\":" + best.map(b => flowJson(b._2)).getOrElse("[]") +
      ",\"sink\":" + primary.map(jsonNode).getOrElse("null") +
      ",\"source\":" + srcNode.map(jsonNode).getOrElse("null") +
      ",\"structures\":" + arr(structs.map(st => "{\"kind\":\"" + st.kind + "\",\"line\":" + st.line + ",\"code\":\"" + esc(st.code.take(160)) + "\"}")) +
      ",\"predicates\":" + arr(preds.map { case (st, r) => "{\"code\":\"" + esc(codeOf(st.pred.get).take(160)) + "\",\"line\":" + st.line + ",\"refs_source\":" + r + "}" }) +
      "}"
  }
  java.nio.file.Files.write(java.nio.file.Paths.get(outPath), payload.getBytes("UTF-8"))
'''


def build_query(import_dir: str, out_path: str, language: str, target_name: str, family_cfg: dict,
                common_sources: dict) -> str:
    mode = "uaf" if family_cfg.get("dealloc_calls") else "leak" if family_cfg.get("acquire_calls") else "standard"
    importer = "cpp" if language == "c++" else "c"
    return "\n".join([
        "import io.shiftleft.semanticcpg.language._",
        "import io.joern.dataflowengineoss.language._",
        "import io.shiftleft.codepropertygraph.generated.nodes._",
        "{",
        "  try { workspace.projects.map(_.name).foreach(n => delete(n)) } catch { case _: Throwable => () }",
        f"  val mycpg = importCode.{importer}({_s(import_dir)})",
        f"  val tname = {_s(target_name)}",
        f"  val outPath = {_s(out_path)}",
        f"  val mode = {_s(mode)}",
        f"  val sinkCallArgs = {_idx_map(family_cfg.get('sink_calls'))}",
        f"  val sinkOpArgs = {_idx_map(family_cfg.get('sink_operators'))}",
        f"  val retSrc = {_str_set(list(common_sources.get('return_calls', [])) + list(family_cfg.get('source_calls', [])))}",
        f"  val outSrc = {_idx_map(common_sources.get('outparam_calls'))}",
        f"  val deallocs = {_str_set(family_cfg.get('dealloc_calls', []))}",
        f"  val acquires = {_str_set(family_cfg.get('acquire_calls', []))}",
        f"  val releases = {_str_set(['free', 'fclose', 'close', 'closedir', 'pclose', 'munmap', '<operator>.delete', '<operator>.deleteArray'])}",
        _BODY,
        "}",
    ])


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class JoernServer:
    """A long-lived ``joern --server`` process (one query at a time)."""

    def __init__(self, joern_home: str, port: int = 0, log_dir: Optional[str] = None,
                 startup_timeout: float = 300.0) -> None:
        launcher = Path(joern_home) / "joern" if joern_home else Path("joern")
        self.launcher = str(launcher)
        self.port = port or _free_port()
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.workdir = tempfile.mkdtemp(prefix="vmavul_joern_")

    def start(self) -> None:
        log = open(os.path.join(self.log_dir or self.workdir, f"joern_{self.port}.log"), "wb")
        self.proc = subprocess.Popen([self.launcher, "--server", "--server-host", "127.0.0.1",
                                      "--server-port", str(self.port)], cwd=self.workdir,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "TERM": "dumb"})
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise JoernError(f"joern server exited during startup (see {log.name})")
            try:
                if self._post("1", 5.0).get("success"):
                    return
            except JoernError:
                time.sleep(2)
        raise JoernError("joern server did not become ready")

    def _post(self, query: str, timeout: float) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/query-sync",
                                     data=json.dumps({"query": query}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as fp:
                return json.loads(fp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise JoernError(f"joern server unreachable: {exc}") from exc

    def query(self, scala: str, timeout: float) -> str:
        with self.lock:
            if self.proc is None:
                self.start()
            resp = self._post(scala, timeout)
        out = _ANSI_RE.sub("", resp.get("stdout", "") or "")
        if not resp.get("success") or "-- [E" in out or "Exception" in out.split("\n", 1)[0]:
            errs = [ln for ln in out.splitlines() if "-- [E" in ln or "|" in ln[:6] or "Exception" in ln]
            raise JoernError("joern query failed: " + "\n".join(errs)[:4000])
        return out

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15)
            except ProcessLookupError:
                pass
        self.proc = None


class JoernPool:
    def __init__(self, joern_home: str, n: int = 1, base_port: int = 0) -> None:
        self.servers = [JoernServer(joern_home, (base_port + i) if base_port else 0) for i in range(max(1, n))]
        self._i = 0
        self._lock = threading.Lock()

    def acquire(self) -> JoernServer:
        with self._lock:
            for s in self.servers:            # prefer an idle server
                if not s.lock.locked():
                    return s
            self._i = (self._i + 1) % len(self.servers)
            return self.servers[self._i]

    def stop(self) -> None:
        for s in self.servers:
            s.stop()


def analyze(pool: JoernPool, unit_code: str, language: str, target_name: str, family_cfg: dict,
            common_sources: dict, timeout: float = 180.0) -> JoernAnalysis:
    with tempfile.TemporaryDirectory(prefix="vmavul_cpg_") as tmp:
        src_dir = Path(tmp) / f"u_{uuid.uuid4().hex[:8]}"
        src_dir.mkdir()
        (src_dir / ("unit.cpp" if language == "c++" else "unit.c")).write_text(unit_code)
        out = Path(tmp) / "result.json"
        pool.acquire().query(build_query(str(src_dir), str(out), language, target_name,
                                         family_cfg, common_sources), timeout)
        if not out.exists():
            raise JoernError("joern analysis produced no result")
        data = json.loads(out.read_text(), strict=False)
    if not data.get("target_found"):
        return JoernAnalysis(False)
    return JoernAnalysis(True, sinks=data.get("sinks", []), sources=data.get("sources", []),
                         taint_edges=data.get("taint_edges"), flow=data.get("flow", []),
                         sink=data.get("sink"), source=data.get("source"),
                         structures=data.get("structures", []), predicates=data.get("predicates", []))
