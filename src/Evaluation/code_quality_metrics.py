import ast
import hashlib
import math
import re
from pathlib import Path
import sys


def evaluate_python_file_quality(py_file: str) -> dict[str, float]:
    """Return static code quality metric scores (0-100) for a Python file.

    The returned dict always has these keys:
    - parse_pass_rate
    - lint_weighted_density
    - type_error_density
    - complexity_score
    - duplication_ratio
    - dead_code_ratio
    - security_risk_density
    - reproducibility_static_score
    - final_score
    """

    path = Path(py_file)
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    loc = max(1, len(lines))
    kloc = loc / 1000.0

    def exp_score(x: float, tau: float) -> float:
        return max(0.0, min(100.0, 100.0 * math.exp(-(x / max(1e-9, tau)))))

    def inv_ratio_score(x: float) -> float:
        return max(0.0, min(100.0, 100.0 * (1.0 - max(0.0, min(1.0, x)))))

    # 1) Parse pass rate for one file.
    try:
        tree = ast.parse(source)
        parse_pass_rate = 100.0
    except SyntaxError:
        return {
            "parse_pass_rate": 0.0,
            "lint_weighted_density": 0.0,
            "type_error_density": 0.0,
            "complexity_score": 0.0,
            "duplication_ratio": 0.0,
            "dead_code_ratio": 0.0,
            "security_risk_density": 0.0,
            "reproducibility_static_score": 0.0,
            "final_score": 0.0,
        }

    # 2) Lint weighted density (lightweight static heuristics).
    lint_err = 0
    lint_warn = 0
    lint_info = 0

    for line in lines:
        if len(line) > 120:
            lint_warn += 1
        if line.rstrip() != line:
            lint_info += 1

    class _LintVisitor(ast.NodeVisitor):
        def visit_Try(self, node: ast.Try) -> None:
            nonlocal lint_warn
            for h in node.handlers:
                if h.type is None:
                    lint_warn += 1
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            nonlocal lint_warn
            if any(alias.name == "*" for alias in node.names):
                lint_warn += 1
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            nonlocal lint_err
            defaults = list(node.args.defaults) + list(node.args.kw_defaults)
            for d in defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    lint_err += 1
            self.generic_visit(node)

    _LintVisitor().visit(tree)
    lwd = (3.0 * lint_err + 2.0 * lint_warn + 1.0 * lint_info) / max(kloc, 1e-9)
    lint_weighted_density = exp_score(lwd, tau=40.0)

    # Collect functions for complexity and typing metrics.
    func_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    n_func = max(1, len(func_nodes))

    # 3) Type error density proxy (annotation risk density).
    type_risks = 0
    for fn in func_nodes:
        public_fn = not fn.name.startswith("_")
        args = list(fn.args.args) + list(fn.args.kwonlyargs)
        if fn.args.vararg is not None:
            args.append(fn.args.vararg)
        if fn.args.kwarg is not None:
            args.append(fn.args.kwarg)

        if public_fn:
            if fn.returns is None:
                type_risks += 1
            for a in args:
                if a.annotation is None:
                    type_risks += 1

    ted = type_risks / max(kloc, 1e-9)
    type_error_density = exp_score(ted, tau=30.0)

    # 4) Cyclomatic overload ratio (sub-metric used by complexity_score).
    def cyclomatic_complexity(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        cc = 1
        for n in ast.walk(fn):
            if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)):
                cc += 1
            elif isinstance(n, ast.BoolOp):
                cc += max(1, len(n.values) - 1)
            elif isinstance(n, ast.comprehension):
                cc += 1
            elif isinstance(n, ast.Match):
                cc += max(1, len(n.cases))
        return cc

    cc_threshold = 10
    cor_raw = sum(max(0, cyclomatic_complexity(f) - cc_threshold) for f in func_nodes) / n_func
    cyclomatic_overload_ratio = exp_score(cor_raw, tau=2.5)

    # 5) Cognitive overload ratio (sub-metric used by complexity_score).
    def cognitive_complexity(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        score = 0

        def walk(node: ast.AST, nesting: int) -> None:
            nonlocal score
            for child in ast.iter_child_nodes(node):
                inc = 0
                next_nesting = nesting
                if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match)):
                    inc = 1 + nesting
                    next_nesting = nesting + 1
                elif isinstance(child, ast.BoolOp):
                    inc = max(1, len(child.values) - 1)
                elif isinstance(child, ast.IfExp):
                    inc = 1 + nesting
                score += inc
                walk(child, next_nesting)

        walk(fn, 0)
        return score

    cg_threshold = 15
    cogor_raw = sum(max(0, cognitive_complexity(f) - cg_threshold) for f in func_nodes) / n_func
    cognitive_overload_ratio = exp_score(cogor_raw, tau=3.0)

    # 6) Long function penalty (sub-metric used by complexity_score).
    total_func_loc = 0
    overflow = 0
    len_threshold = 60
    for f in func_nodes:
        end_lineno = getattr(f, "end_lineno", f.lineno)
        f_loc = max(1, end_lineno - f.lineno + 1)
        total_func_loc += f_loc
        overflow += max(0, f_loc - len_threshold)
    lfp_raw = overflow / max(1, total_func_loc)
    long_function_penalty = inv_ratio_score(lfp_raw)

    # Merge 4/5/6 into one unified complexity metric.
    complexity_score = (
        cyclomatic_overload_ratio + cognitive_overload_ratio + long_function_penalty
    ) / 3.0

    # 7) Duplication ratio via normalized 5-line windows.
    window = 5
    normalized = [re.sub(r"\s+", "", ln) for ln in lines]
    hashes: dict[str, int] = {}
    for i in range(0, max(0, len(normalized) - window + 1)):
        chunk = "\n".join(normalized[i : i + window])
        if not chunk.strip():
            continue
        h = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
        hashes[h] = hashes.get(h, 0) + 1
    duplicated_windows = sum(c - 1 for c in hashes.values() if c > 1)
    duplicated_loc = duplicated_windows * window
    dr_raw = duplicated_loc / max(1, loc)
    duplication_ratio = inv_ratio_score(dr_raw)

    # 8) Dead code ratio (imports/variables/functions never used).
    imported_names = set()
    assigned_names = set()
    used_names = set()
    defined_functions = set()
    called_functions = set()

    class _DeadCodeVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imported_names.add((alias.asname or alias.name.split(".")[0]))
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    imported_names.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        assigned_names.add(n.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            t = node.target
            if isinstance(t, ast.Name):
                assigned_names.add(t.id)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            defined_functions.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            defined_functions.add(node.name)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                called_functions.add(node.func.id)
            self.generic_visit(node)

    _DeadCodeVisitor().visit(tree)
    unused_imports = len([n for n in imported_names if n not in used_names and not n.startswith("_")])
    unused_vars = len([n for n in assigned_names if n not in used_names and not n.startswith("_")])
    unused_funcs = len([n for n in defined_functions if n not in called_functions and n != "main" and not n.startswith("_")])
    symbol_count = max(1, len(imported_names) + len(assigned_names) + len(defined_functions))
    dcr_raw = (unused_imports + unused_vars + unused_funcs) / symbol_count
    dead_code_ratio = inv_ratio_score(dcr_raw)

    # 9) Security risk density.
    sec_critical = 0
    sec_high = 0
    sec_medium = 0

    class _SecurityVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal sec_critical, sec_high, sec_medium

            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in {"eval", "exec"}:
                sec_critical += 1

            if func_name in {"system", "popen"}:
                sec_high += 1

            if func_name == "run":
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        sec_high += 1

            if func_name == "load":
                # yaml.load(..., Loader=...) not counted unless missing safe loader.
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in {"yaml", "pickle"}:
                        if node.func.value.id == "pickle":
                            sec_high += 1
                        else:
                            has_loader = any(kw.arg == "Loader" for kw in node.keywords)
                            if not has_loader:
                                sec_medium += 1

            self.generic_visit(node)

    _SecurityVisitor().visit(tree)

    secret_patterns = [
        re.compile(r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]"),
        re.compile(r"(?i)secret\s*=\s*['\"][^'\"]{8,}['\"]"),
        re.compile(r"(?i)password\s*=\s*['\"][^'\"]{6,}['\"]"),
    ]
    for ln in lines:
        if any(p.search(ln) for p in secret_patterns):
            sec_high += 1

    srd = (5.0 * sec_critical + 3.0 * sec_high + 1.0 * sec_medium) / max(kloc, 1e-9)
    security_risk_density = exp_score(srd, tau=15.0)

    # 10) Reproducibility static score (weighted checklist).
    checks = []

    has_seed = bool(
        re.search(r"random_state\s*=\s*\d+", source)
        or re.search(r"np\.random\.seed\s*\(", source)
        or re.search(r"random\.seed\s*\(", source)
    )
    checks.append((2.0, has_seed))

    has_split_seed = bool(re.search(r"train_test_split\(.*random_state\s*=", source, flags=re.DOTALL))
    checks.append((2.0, has_split_seed))

    # Penalize hardcoded absolute paths.
    has_abs_path = bool(re.search(r"['\"][A-Za-z]:\\\\|['\"]/[^'\"]+", source))
    checks.append((2.0, not has_abs_path))

    uses_config = bool(re.search(r"(yaml|json|toml|argparse|typer|dotenv|os\.environ)", source))
    checks.append((1.0, uses_config))

    saves_artifact = bool(re.search(r"(joblib\.dump|pickle\.dump|to_csv\(|save\()", source))
    checks.append((1.0, saves_artifact))

    has_version_pin_hint = bool(re.search(r"(__version__|requirements|pyproject|poetry)", source, flags=re.IGNORECASE))
    checks.append((1.0, has_version_pin_hint))

    num = sum(w for w, ok in checks if ok)
    den = sum(w for w, _ in checks)
    reproducibility_static_score = max(0.0, min(100.0, 100.0 * (num / max(1e-9, den))))

    component_scores = [
        parse_pass_rate,
        lint_weighted_density,
        type_error_density,
        complexity_score,
        duplication_ratio,
        dead_code_ratio,
        security_risk_density,
        reproducibility_static_score,
    ]
    final_score = sum(component_scores) / len(component_scores)

    return {
        "parse_pass_rate": round(parse_pass_rate, 4),
        "lint_weighted_density": round(lint_weighted_density, 4),
        "type_error_density": round(type_error_density, 4),
        "complexity_score": round(complexity_score, 4),
        "duplication_ratio": round(duplication_ratio, 4),
        "dead_code_ratio": round(dead_code_ratio, 4),
        "security_risk_density": round(security_risk_density, 4),
        "reproducibility_static_score": round(reproducibility_static_score, 4),
        "final_score": round(final_score, 4),
    }

# 如果直接运行这个脚本：
# 1) 可用命令行参数：python code_quality_metrics.py path/to/file.py
# 2) 也可无参数运行后交互输入路径
if __name__ == "__main__":
    import json

    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter the path to the Python file to evaluate: ").strip()
        if not input_file:
            print("Error: empty file path.")
            print("Usage: python code_quality_metrics.py <python_file.py>")
            sys.exit(1)

    if not Path(input_file).exists():
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    metrics = evaluate_python_file_quality(input_file)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))