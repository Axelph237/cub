import json

from cub import Bool, Int, Struct, UInt, to_json

I8 = {"kind": "bv", "width": 8, "signed": True}
BOOL = {"kind": "bool"}
BITS8 = {"kind": "struct", "elements": [BOOL] * 8}


def bits(width: int) -> dict[str, object]:
    return {"kind": "struct", "elements": [BOOL] * width}


def test_two_variable_constraint_matches_the_format_contract() -> None:
    x, y = Int.var("x", 8), Int.var("y", 8)
    assert json.loads(to_json(x + y == 3)) == {
        "version": 2,
        "nodes": [
            {"id": 0, "op": "var", "name": "x", "type": BITS8},
            {"id": 1, "op": "view", "args": [0], "signed": True, "type": I8},
            {"id": 2, "op": "var", "name": "y", "type": BITS8},
            {"id": 3, "op": "view", "args": [2], "signed": True, "type": I8},
            {"id": 4, "op": "add", "args": [1, 3], "type": I8},
            {"id": 5, "op": "const", "value": 3, "type": I8},
            {"id": 6, "op": "eq", "args": [4, 5], "type": BOOL},
        ],
        "constraints": [6],
    }


def test_shift_exports_its_unsigned_amount() -> None:
    x = Int.var("x", 8)
    assert json.loads(to_json(x >> 3 == -1)) == {
        "version": 2,
        "nodes": [
            {"id": 0, "op": "var", "name": "x", "type": BITS8},
            {"id": 1, "op": "view", "args": [0], "signed": True, "type": I8},
            {"id": 2, "op": "const", "value": 3, "type": {"kind": "bv", "width": 8, "signed": False}},
            {"id": 3, "op": "shr", "args": [1, 2], "type": I8},
            {"id": 4, "op": "const", "value": -1, "type": I8},
            {"id": 5, "op": "eq", "args": [3, 4], "type": BOOL},
        ],
        "constraints": [5],
    }


def test_shared_subexpression_is_emitted_once() -> None:
    x, y = Int.var("x", 8), Int.var("y", 8)
    total = x + y
    exported = json.loads(to_json(total * total == 1, x + y < 5))
    adds = [node for node in exported["nodes"] if node["op"] == "add"]
    assert len(adds) == 1, "x + y built twice is one structural node"
    (mul,) = [node for node in exported["nodes"] if node["op"] == "mul"]
    assert mul["args"] == [adds[0]["id"]] * 2


def test_every_node_kind_exports_its_fields() -> None:
    u, p = UInt.var("u", 4), Bool.var("p")
    constraint = ~p | (p.ite(-u, ~u).zext(6).trunc(5).as_signed().sext(7) != 0)
    ops = {node["op"]: node for node in json.loads(to_json(constraint))["nodes"]}
    assert set(ops) == {
        "var", "view", "neg", "bvnot", "ite", "zext", "trunc", "reinterpret", "sext",
        "const", "ne", "not", "or",
    }
    assert (ops["zext"]["width"], ops["sext"]["width"]) == (6, 7)
    assert ops["sext"]["type"] == {"kind": "bv", "width": 7, "signed": True}


def test_nested_struct_exports_every_struct_op() -> None:
    p, s = Bool.var("p"), Struct.var("s", 2, of=2)
    constraint = (s.sub(0) == Struct(p, p)) | (s.flatten().as_uint().struct.as_int() == 0)
    assert json.loads(to_json(constraint)) == {
        "version": 2,
        "nodes": [
            {"id": 0, "op": "var", "name": "s", "type": {"kind": "struct", "elements": [bits(2), bits(2)]}},
            {"id": 1, "op": "index", "args": [0], "index": 0, "type": bits(2)},
            {"id": 2, "op": "var", "name": "p", "type": BOOL},
            {"id": 3, "op": "pack", "args": [2, 2], "type": bits(2)},
            {"id": 4, "op": "eq", "args": [1, 3], "type": BOOL},
            {"id": 5, "op": "flatten", "args": [0], "type": bits(4)},
            {"id": 6, "op": "view", "args": [5], "signed": False, "type": {"kind": "bv", "width": 4, "signed": False}},
            {"id": 7, "op": "bits", "args": [6], "type": bits(4)},
            {"id": 8, "op": "view", "args": [7], "signed": True, "type": {"kind": "bv", "width": 4, "signed": True}},
            {"id": 9, "op": "const", "value": 0, "type": {"kind": "bv", "width": 4, "signed": True}},
            {"id": 10, "op": "eq", "args": [8, 9], "type": BOOL},
            {"id": 11, "op": "or", "args": [4, 10], "type": BOOL},
        ],
        "constraints": [11],
    }


def test_count_comparison_exports_the_count_op() -> None:
    u2 = {"kind": "bv", "width": 2, "signed": False}
    assert json.loads(to_json(Struct.var("s", 3).count() <= 1)) == {
        "version": 2,
        "nodes": [
            {"id": 0, "op": "var", "name": "s", "type": bits(3)},
            {"id": 1, "op": "count", "args": [0], "type": u2},
            {"id": 2, "op": "const", "value": 1, "type": u2},
            {"id": 3, "op": "le", "args": [1, 2], "type": BOOL},
        ],
        "constraints": [3],
    }
