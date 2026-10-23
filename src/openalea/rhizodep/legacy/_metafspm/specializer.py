import ast, inspect, textwrap, linecache
import numpy as np
from numba import njit


# ------------------- AST helpers -------------------

def _parents_map(tree):
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    return parents

def _infer_attrs_to_inline(fdef):
    parents = _parents_map(fdef)
    reads, mutated = set(), set()
    for n in ast.walk(fdef):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self':
            if isinstance(n.ctx, ast.Store):
                mutated.add(n.attr)
            else:
                p = parents.get(n)
                if isinstance(p, ast.Call) and p.func is n:  # it's a call target -> skip
                    continue
                reads.add(n.attr)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute):
            a = n.value
            if (isinstance(a.value, ast.Name) and a.value.id == 'self' and
                isinstance(n.ctx, ast.Store)):
                mutated.add(a.attr)
    return reads - mutated

def _find_self_callees(fdef):
    names = set()
    for n in ast.walk(fdef):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if isinstance(n.func.value, ast.Name) and n.func.value.id == 'self':
                names.add(n.func.attr)
    return names

def _stable_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]

class StripDecorators(ast.NodeTransformer):
    """Remove all decorators from functions/classes in the parsed tree."""
    def visit_FunctionDef(self, node):
        node.decorator_list = []          # drop @dec(...)
        self.generic_visit(node)
        return node
    def visit_AsyncFunctionDef(self, node):
        node.decorator_list = []
        self.generic_visit(node)
        return node
    def visit_ClassDef(self, node):
        node.decorator_list = []          # in case you ever parse a class block
        self.generic_visit(node)
        return node

# ------------------- Recursive specializer -------------------

def specialize_method_recursive(method, instance, max_depth=2, registry=None, print_src=False, debug=False):
    """
    Recursively specialize `method` and its nested self.method(...) callees.
    Returns (numba_dispatcher, registry).
    """
    if registry is None:
        registry = {}
    cls = instance.__class__
    visiting = set()

    def _specialize_by_name(name: str, depth: int):
        if name in registry:
            return
        if name in visiting:
            if debug: print(f"Cyclic method calls detected at '{name}'")
        if depth < 0:
            if debug: print(f"Max recursion depth exceeded while specializing '{name}'")

        py_method = getattr(cls, name)  # unbound function
        src = textwrap.dedent(inspect.getsource(py_method))
        tree = ast.parse(src)

        # Remove decorators
        StripDecorators().visit(tree)
        ast.fix_missing_locations(tree)

        fdef = next(n for n in tree.body if isinstance(n, ast.FunctionDef))

        # 1) recurse into nested self.method(...) first
        callees = _find_self_callees(fdef)
        visiting.add(name)
        for callee in sorted(callees):
            if hasattr(cls, callee) and inspect.isfunction(getattr(cls, callee)):
                _specialize_by_name(callee, depth-1)
        visiting.remove(name)

        # 2) inline self.<attr>
        const_map, global_arrays = {}, {}
        for attr in sorted(_infer_attrs_to_inline(fdef)):
            val = getattr(instance, attr)
            if isinstance(val, (bool, int, float, np.bool_, np.integer, np.floating)):
                const_map[attr] = bool(val) if isinstance(val, np.bool_) \
                                  else int(val) if isinstance(val, np.integer) \
                                  else float(val) if isinstance(val, np.floating) \
                                  else val
            elif isinstance(val, np.ndarray):
                global_arrays[attr] = val  # inject as read-only global
            elif isinstance(val, tuple) and all(isinstance(x, (bool, int, float, np.bool_, np.integer, np.floating)) for x in val):
                const_map[attr] = tuple(bool(x) if isinstance(x, np.bool_) else int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x
                                        for x in val)

        sym_for = {callee: f"__HELP_{callee}" for callee in callees}

        class Rewriter(ast.NodeTransformer):
            def visit_Attribute(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == 'self':
                    nm = node.attr
                    if nm in const_map:
                        return ast.copy_location(ast.Constant(const_map[nm]), node)
                    if nm in global_arrays:
                        return ast.copy_location(ast.Name(id=f'__C_{nm}', ctx=ast.Load()), node)
                return node
            def visit_Call(self, node):
                self.generic_visit(node)
                f = node.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == 'self' and f.attr in sym_for):
                    node.func = ast.copy_location(ast.Name(id=sym_for[f.attr], ctx=ast.Load()), f)
                return node

        Rewriter().visit(tree)
        ast.fix_missing_locations(tree)

        # 3) drop 'self'
        if fdef.args.args and fdef.args.args[0].arg == 'self':
            fdef.args.args = fdef.args.args[1:]
        fdef.name = name

        # 4) pretty source (optional)
        gen_src = ast.unparse(tree)
        if print_src:
            print(f"\n=== specialized {name} ===\n{gen_src}")

        # keep inspect.getsource() working
        fake_fn = f"<specialized:{id(instance)}:{name}:{_stable_hash(gen_src)}>"
        linecache.cache[fake_fn] = (len(gen_src), None, gen_src.splitlines(True), fake_fn)

        # 5) exec globals: start from method's module globals (so np & other helpers are visible)
        glb = dict(py_method.__globals__)
        glb.setdefault('np', np)
        for an, arr in global_arrays.items():
            glb[f"__C_{an}"] = arr
        for callee in callees:
            if callee in registry:
                glb[sym_for[callee]] = registry[callee]['jit']  # compiled helper

        # 6) exec + JIT
        try:
            ns = {}
            code = compile(tree, fake_fn, "exec")
            exec(code, glb, ns)
            py_func = ns[name]
            jitted = njit(py_func, cache=False)

            # try compile once (ignore arg mismatch)
            try:
                _ = jitted(*(1 for _ in range(len(fdef.args.args))))
                registry[name] = {'jit': jitted, 'py': py_func, 'src': gen_src}

            except Exception:
                pass

        except Exception:
            pass

    entry_name = method.__name__ if inspect.ismethod(method) else method.__name__
    _specialize_by_name(entry_name, max_depth)
    # Debug print 
    # print(registry)
    if entry_name in registry:
        return registry[entry_name]['jit'], registry
    else:
        return None, None