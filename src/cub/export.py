"""Export constraints as a JSON DAG: each distinct node appears once, after its arguments."""

from __future__ import annotations

import json
from collections.abc import Callable

from cub import ir
from cub.dsl import Bool

type Json = dict[str, object]


def to_json(*constraints: Bool) -> str:
    ids: dict[ir.Node, int] = {}
    nodes: list[Json] = []

    def visit(node: ir.Node) -> int:
        if node in ids:
            return ids[node]
        fields = _fields(node, visit)
        ids[node] = len(nodes)
        nodes.append({"id": ids[node], **fields, "type": _type(node.type)})
        return ids[node]

    roots = [visit(constraint.node) for constraint in constraints]
    return json.dumps({"version": 1, "nodes": nodes, "constraints": roots})


def _fields(node: ir.Node, visit: Callable[[ir.Node], int]) -> Json:
    match node:
        case ir.Var(name=name):
            return {"op": "var", "name": name}
        case ir.Const(value=value):
            return {"op": "const", "value": value}
        case ir.Unary(op=op, x=x):
            return {"op": op, "args": [visit(x)]}
        case ir.Binary(op=op, a=a, b=b) | ir.Compare(op=op, a=a, b=b):
            return {"op": op, "args": [visit(a), visit(b)]}
        case ir.Logic(op=op, args=args):
            return {"op": op, "args": [visit(arg) for arg in args]}
        case ir.Not(x=x):
            return {"op": "not", "args": [visit(x)]}
        case ir.Ite(cond=cond, then=then, else_=else_):
            return {"op": "ite", "args": [visit(cond), visit(then), visit(else_)]}
        case ir.Resize(op=op, x=x, width=width):
            return {"op": op, "args": [visit(x)], "width": width}


def _type(type_: ir.Type) -> Json:
    match type_:
        case ir.BV(width=width, signed=signed):
            return {"kind": "bv", "width": width, "signed": signed}
        case ir.BoolType():
            return {"kind": "bool"}
