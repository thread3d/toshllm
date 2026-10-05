# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool definitions. The schemas here are the only description of the API.

Every word below is paid for in each conversation that offers the tools, so the
definitions say only what a model needs to pick a tool and fill it in. "accepted" lists
arguments the helper takes without advertising them.
"""

_MATH = {"type": "string"}
_LIST = {"type": "array", "items": _MATH}
_COMMON = ("assumptions", "precision", "substitutions")

TOOLS = {
    "expression": {
        "description": "Exact algebra and calculus on one expression. Math is plain text: x**2*sin(x), "
                       "sqrt(2), exp(-t), pi, E, oo.",
        "operations": [
            "simplify", "expand", "factor", "cancel", "together", "apart", "evaluate",
            "differentiate", "integrate", "limit", "series", "summation", "product",
            "laplace_transform", "inverse_laplace_transform",
            "fourier_transform", "inverse_fourier_transform",
        ],
        "properties": {
            "expression": _MATH,
            "variable": _MATH,
            "order": {"type": "integer", "description": "Derivative order or series terms."},
            "lower": _MATH,
            "upper": _MATH,
            "point": {**_MATH, "description": "Limit or series point, e.g. 0 or oo."},
            "direction": {"type": "string", "enum": ["both", "+", "-"]},
            "substitutions": {"type": "object"},
            "assumptions": {"type": "object", "description": "{name: [positive, real, integer]}"},
            "precision": {"type": "integer"},
        },
        "accepted": ("transform_variable",),
        "required": ["operation", "expression"],
    },
    "solve": {
        "description": "Solve equations, systems, inequalities and differential equations (dsolve, written "
                       "with y' and y''). An equation is text like \"x**2 - 5*x + 6 = 0\".",
        "operations": ["solve", "solveset", "nsolve", "dsolve"],
        "properties": {
            "equations": _LIST,
            "variables": _LIST,
            "domain": {"type": "string", "enum": ["complex", "real"]},
            "function": _MATH,
            "variable": _MATH,
            "initial_conditions": {"type": "object", "description": "{\"y(0)\": \"1\"}"},
            "initial_guess": {**_LIST, "description": "nsolve: one per variable."},
            "precision": {"type": "integer"},
        },
        "accepted": ("assumptions",),
        "required": ["operation", "equations"],
    },
    "matrix": {
        "description": "Exact linear algebra on a matrix given as rows. linear_solve solves matrix * x = other.",
        "operations": [
            "determinant", "inverse", "transpose", "rank", "rref", "nullspace",
            "eigenvalues", "eigenvectors", "multiply", "linear_solve",
        ],
        "properties": {
            "matrix": {"type": "array", "description": "[[\"1\", \"2\"], [\"3\", \"4\"]]"},
            "other": {"type": "array"},
        },
        "accepted": ("assumptions", "precision"),
        "required": ["operation", "matrix"],
    },
    "verify": {
        "description": "Confirm math or a proposed answer. equivalent: are left and right equal or identical "
                       "expressions. solution: do the values in solution satisfy the equations. Returns the "
                       "difference or residual.",
        "operations": ["equivalent", "solution"],
        "properties": {
            "left": _MATH,
            "right": _MATH,
            "equations": {**_LIST, "description": "solution: the equations to check."},
            "solution": {"type": "object", "description": "{\"x\": \"2\"}"},
            "function": {**_MATH, "description": "ODE: the unknown function, e.g. y."},
        },
        "accepted": ("variable", "assumptions", "precision", "substitutions"),
        "required": ["operation"],
    },
}


def definitions():
    """The tools in MCP tools/list form."""
    listed = []
    for name, tool in TOOLS.items():
        properties = {"operation": {"type": "string", "enum": tool["operations"]}}
        properties.update(tool["properties"])
        listed.append({
            "name": name,
            "description": tool["description"],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": tool["required"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        })
    return listed


def check(name, arguments):
    """Returns the operation, or raises ValueError naming what is wrong with the call."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"unknown tool '{str(name)[:40]}'")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    operation = arguments.get("operation")
    if operation not in tool["operations"]:
        raise ValueError("'operation' must be one of: " + ", ".join(tool["operations"]))
    allowed = set(tool["properties"]) | set(tool["accepted"]) | set(_COMMON)
    unknown = [key for key in arguments if key != "operation" and key not in allowed]
    if unknown:
        raise ValueError(f"unknown argument '{str(unknown[0])[:40]}'; allowed: " + ", ".join(tool["properties"]))
    return operation
