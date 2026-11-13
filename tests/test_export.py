import json

from cub import Bool, Int, UInt, to_json

I8 = {"kind": "bv", "width": 8, "signed": True}


def test_two_variable_constraint_matches_the_format_contract() -> None:
    x, y = Int.var("x", 8), Int.var("y", 8)
    assert json.loads(to_json(x + y == 3)) == {
        "version": 1,
        "nodes": [
            {"id": 0, "op": "var", "name": "x", "type": I8},
            {"id": 1, "op": "var", "name": "y", "type": I8},
            {"id": 2, "op": "add", "args": [0, 1], "type": I8},
            {"id": 3, "op": "const", "value": 3, "type": I8},
            {"id": 4, "op": "eq", "args": [2, 3], "type": {"kind": "bool"}},
        ],
        "constraints": [4],
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
        "var", "neg", "bvnot", "ite", "zext", "trunc", "reinterpret", "sext",
        "const", "ne", "not", "or",
    }
    assert (ops["zext"]["width"], ops["sext"]["width"]) == (6, 7)
    assert ops["sext"]["type"] == {"kind": "bv", "width": 7, "signed": True}
