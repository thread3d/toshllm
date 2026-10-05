# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The rules a math turn follows, with no I/O: what a request asks of the tools, when a turn
gives up after refused calls, which results count as validated, and which numbers an answer
may state. The agent applies them; nothing here talks to the engine.
"""

import json
import math
import re

MATH_PREFIXES = ("sympy_", "scientific_")
CLARIFY = "ask_user_to_clarify"
REJECTIONS = {"transcription_mismatch", "needs_review"}


def is_math(name):
    return isinstance(name, str) and name.startswith(MATH_PREFIXES)


def reply_of(text):
    """The JSON reply inside a tool result, which may carry a prefix such as "error: "."""
    if not isinstance(text, str) or "{" not in text:
        return None
    try:
        value = json.loads(text[text.index("{"):])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def error_code(reply):
    error = (reply or {}).get("error")
    return error.get("code") if isinstance(error, dict) else None


# what the request asks

NO_MATH, CONCEPTUAL, COMPUTATIONAL, AMBIGUOUS = "no_math", "conceptual", "computational", "ambiguous"
INTENTS = (NO_MATH, CONCEPTUAL, COMPUTATIONAL, AMBIGUOUS)


def requires_tools(intent):
    return intent in (COMPUTATIONAL, AMBIGUOUS)


def _rx(pattern):
    return re.compile(pattern, re.I)


_LATEX = _rx(r"\\(?:frac|dfrac|int|iint|oint|sum|prod|sqrt|lim|partial|nabla|infty|cdot|times|begin\{[pbvB]?matrix\}"
             r"|left|right|mathrm|alpha|beta|gamma|lambda|mu|sigma|theta|omega|pi)(?![A-Za-z])|\$[^$\n]+\$")
_UNICODE = _rx(r"[∫∬∮∑∏√∞≤≥≠≈±∂∇∆×÷²³⁴⁵⁶⁷⁸⁹ⁿ₀₁₂₃∈∉⊂∪∩]")
_EQUATION = _rx(r"(?:[A-Za-z]\w*(?:\([^()]*\))?|\d)\s*(?:'+\s*)?(?:=|<=|>=|<|>)\s*[-+(]?\s*(?:\d|[A-Za-z]\b)")
_OPERATION = _rx(r"(?:\b\d+(?:[.,]\d+)?|\b[A-Za-z]\b|\))\s*(?:\^|\*\*|[*/+−-])\s*(?:\d|\b[A-Za-z]\b|\()|\b\d+[A-Za-z]\b(?!\w)")
_FUNCTION = _rx(r"\b(?:sin|cos|tan|exp|log|ln|sqrt|abs|sinh|cosh|tanh|arctan|atan|erf|gamma|zeta)\s*\(")
_DERIVATIVE = _rx(r"\bd\^?\d?[A-Za-z]\s*/\s*d[A-Za-z]|\b[A-Za-z]'{1,3}\s*(?:\(|[-+=])|∂")
_MATRIX = _rx(r"\[\s*\[|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*,|\\begin\{")
_LIST = _rx(r"-?\d+(?:[.,]\d+)?(?:\s*[,;]\s*-?\d+(?:[.,]\d+)?){3,}")
_CODE = _rx(r"```|\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(|\breturn\b|\bprint\s*\(|\bimport\s+\w+|console\.log|=>|\bclass\s+\w+")
_PRECISION = _rx(r"\b\d+\s*(?:decimal|decimales|cifras|d[ií]gitos|digits|significant|significativas|places)\b"
                 r"|\b(?:tolerance|tolerancia|precisi[oó]n|precision|accuracy)\s*(?:of|de|=|:)?\s*\d|\bto\s+\d+\s+(?:dp|sf)\b")
# short stems on purpose: they match the forms of a verb in both languages
_COMPUTE = _rx(r"\b(?:calcul|comput|solv|resuelv|resolv|evalu|eval[uú]|integr|deriv|differentiat|diferenci|simplif|factori"
               r"|expand|desarroll|halla|hallar|find|determin[ae]|obt[eé]n|obtain|verif|comprueb|check|aproxim|approximat"
               r"|ajust|fit|regres|interpol|minimi|maximi|optimi|estim|cu[aá]nto|cu[aá]nta|how much|how many"
               r"|what is the value|cu[aá]l es el valor|value of|valor de|result|resultado|give me|dame|dime|tell me"
               r"|ra[ií]ces de|roots? of|zeros? of|ceros de|probabilidad de que|probability that)")
_OBJECT = _rx(r"\b(?:ecuaci[oó]n diferencial|differential equation|integral|derivad|derivative|diferencial|ecuaci[oó]n|equation|sistema de|system of|matri[zx]|matrices|determinant"
              r"|autovalor|eigen|valor(?:es)? propio|inversa|inverse|l[ií]mite|limit|serie|series|sumatori|ra[ií]z|root|fft"
              r"|fourier|laplace|se[nñ]al|signal|filtro|filter|regresi[oó]n|regression|media|mean|mediana|median|promedio"
              r"|average|desviaci[oó]n|deviation|varianza|variance|percentil|percentile|probabilidad|probability"
              r"|distribuci[oó]n|distribution|edo|ode|ecuaci[oó]n diferencial|differential equation|transformada|transform"
              r"|polinomio|polynomial|vector|producto escalar|dot product|cross product|producto vectorial|svd"
              r"|valores singulares|singular value|m[ií]nimos cuadrados|least squares|t-test|prueba t|chi|correlaci[oó]n"
              r"|correlation|interpolaci[oó]n|interpolation|gradiente|gradient|hessian|jacobian|optimizaci[oó]n"
              r"|optimization|m[ií]nimo|minimum|m[aá]ximo|maximum|logaritmo|logarithm|seno|coseno|tangente|sine|cosine"
              r"|tangent|pi\b|π|infinit)")
_CONCEPT = _rx(r"\b(?:qu[eé] es|qu[eé] son|what is an?\b|what are|what's an?\b|expl[ií]ca|explain|por qu[eé]|why"
               r"|c[oó]mo funciona|how does|how do|qu[eé] significa|what does .{1,40}(?:mean|represent)|qu[eé] representa"
               r"|represent|diferencia entre|difference between|intuici[oó]n|intuition|para qu[eé] sirve"
               r"|what is .{1,30} used for|useful|[uú]til|defin|concept|concepto|meaning|significado|historia|history"
               r"|ejemplo de|example of|describe|describ)")
_DIGIT = re.compile(r"\d")
_VARIABLE = _rx(r"\b(?:x|y|z|t|n)\b")


def evidence(text):
    text = str(text or "")
    structure = any(p.search(text) for p in (_LATEX, _UNICODE, _EQUATION, _OPERATION, _FUNCTION, _DERIVATIVE, _MATRIX, _LIST))
    # "integral" names an object and "diferencia entre" asks for an explanation: neither asks to compute
    verbs = _CONCEPT.sub(" ", _OBJECT.sub(" ", text))
    return {
        "code": bool(_CODE.search(text)),
        "structure": structure,
        "concrete": bool(_DIGIT.search(text)) or structure or bool(_VARIABLE.search(text)),
        "compute": bool(_COMPUTE.search(verbs)),
        "object": bool(_OBJECT.search(text)),
        "concept": bool(_CONCEPT.search(text)),
        "precision": bool(_PRECISION.search(text)),
    }


def _strong_structure(text):
    return bool(_LATEX.search(text) or re.search(r"[∫∑∏√]", text) or _MATRIX.search(text)
                or (_EQUATION.search(text) and re.search(r"\b[a-z]\b", text)))


def classify(text, context=()):
    """The deterministic reading of a request, or None when only a model can tell."""
    e = evidence(text)
    earlier = any(evidence(c)["structure"] for c in context)
    if e["code"] and not e["compute"] and not e["precision"]:
        return NO_MATH
    if not e["structure"] and not e["object"] and not e["precision"]:
        return None if e["compute"] else NO_MATH
    if e["precision"] and (e["structure"] or e["object"]):
        return COMPUTATIONAL
    if e["structure"] and e["compute"] and not e["concept"]:
        return COMPUTATIONAL
    if e["structure"] and not e["compute"] and not e["concept"]:
        return COMPUTATIONAL if _strong_structure(text) else None
    if e["structure"]:
        return None
    if e["concept"] and not e["compute"]:
        return None if e["concrete"] else CONCEPTUAL
    if e["compute"]:
        return None if e["concrete"] or earlier else AMBIGUOUS
    return None


INTENT_INSTRUCTIONS = (
    "Classify the user's message for a math assistant. Answer with one word. computational: it asks for a specific "
    "mathematical or scientific result, such as a value, a solution, a simplification, a fit or a statistic, even "
    "inside prose or an explanation. conceptual: it asks to explain, define or compare ideas and needs no specific "
    "result. ambiguous: it asks for a calculation but leaves out what the calculation needs. no_math: anything else, "
    "including numbers that are not to be computed.")
INTENT_GRAMMAR = 'root ::= "computational" | "conceptual" | "ambiguous" | "no_math"'

REVIEW_INSTRUCTIONS = (
    "You check a transcription. Do not solve anything. REQUEST is what the user asked. CALL is what will be computed. "
    "Answer consistent if CALL states the same mathematical problem as REQUEST: the same formulas, numbers, limits, "
    "conditions and data, even when written differently or with other variable names. Answer inconsistent if CALL "
    "changes, drops or adds any of them. Answer uncertain if REQUEST does not say enough to tell.")
REVIEW_GRAMMAR = 'root ::= "consistent" | "inconsistent" | "uncertain"'


# a turn after refused calls

class Guard:
    """One corrected call is allowed after a refusal; a second refusal ends the turn."""

    def __init__(self):
        self.rejections = 0
        self.pending = False

    @property
    def next(self):
        return "free" if not self.pending else "stop" if self.rejections >= 2 else "math_only"

    def record(self, tool, reply):
        if not is_math(tool) or reply is None:
            return
        if reply.get("success") is True:
            self.pending = False
        elif error_code(reply) in REJECTIONS:
            self.rejections += 1
            self.pending = True

    def closing(self, names, arguments, lang="en"):
        if not self.pending:
            return None
        if CLARIFY in names:
            return unresolved(_missing(arguments[names.index(CLARIFY)]), lang)
        return None if any(is_math(n) for n in names) else unresolved(lang=lang)


def _missing(arguments):
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        return None
    return value.get("missing") if isinstance(value, dict) else None


_SPANISH = re.compile(r"[¿¡ñáéíóú]|\b(?:el|la|los|las|de|del|que|qué|con|para|por|una?|es|calcula|resuelve|halla|dame"
                      r"|integral de|cu[aá]nto|y|en)\b", re.I)
_ENGLISH = re.compile(r"\b(?:the|of|and|what|is|are|with|for|to|calculate|solve|find|give|compute|how)\b", re.I)


def language(texts):
    """The language of the texts Tosh writes itself. English unless the user's own words are clearly Spanish."""
    text = " ".join(str(t or "") for t in texts)[-4000:]
    spanish, english = len(_SPANISH.findall(text)), len(_ENGLISH.findall(text))
    return "es" if spanish >= 2 and spanish > 2 * english else "en"


_TEXTS = {
    "unresolved": {"en": "I couldn't validate the mathematical interpretation of that request. Please rephrase it or "
                         "provide the expression explicitly.",
                   "es": "No pude validar cómo interpretar matemáticamente la petición. Reformúlala o escribe la "
                         "expresión explícitamente."},
    "formula": {"en": "Write out the complete formula or equation.", "es": "Escribe la fórmula o ecuación completa."},
    "data": {"en": "Give every data value or sample.", "es": "Indica todos los datos o muestras."},
    "limits": {"en": "Give the limits or the interval.", "es": "Indica los límites o el intervalo."},
    "conditions": {"en": "Give the initial conditions, if there are any.",
                   "es": "Indica las condiciones iniciales, si las hay."},
    "method": {"en": "Say which method you want.", "es": "Indica el método que quieres usar."},
    "header": {"en": "This is what the tools validated:", "es": "Esto es lo que se pudo validar con las herramientas:"},
    "exact": {"en": "exact", "es": "exacto"},
    "approximate": {"en": "approximate", "es": "aproximado"},
    "no_exact": {"en": "No exact result could be validated: the values above are numerical approximations.",
                 "es": "Ningún resultado exacto se pudo validar: los valores de arriba son aproximaciones numéricas."},
    "open": {"en": "No exact form was obtained: SymPy found no closed form for `{}`.",
             "es": "No se obtuvo una forma exacta: SymPy no encontró forma cerrada para `{}`."},
    "refused": {"en": "Some calculations were not run because they did not match the request.",
                "es": "Algunos cálculos no se hicieron porque no correspondían a la petición."},
    "rest": {"en": "The rest of the request could not be validated with the available tools, and is not filled in from memory.",
             "es": "El resto de lo pedido no se pudo validar con las herramientas disponibles, y no se completa de memoria."},
}


def text(key, lang):
    return _TEXTS[key].get(lang, _TEXTS[key]["en"])


def unresolved(missing=None, lang="en"):
    base = text("unresolved", lang)
    return base + " " + text(missing, lang) if missing in ("formula", "data", "limits", "conditions", "method") else base


CLARIFY_TOOL = {
    "type": "function",
    "function": {
        "name": CLARIFY,
        "description": "Ask the user to restate the problem when the math call cannot be written from the request.",
        "parameters": {"type": "object",
                       "properties": {"missing": {"type": "string",
                                                  "enum": ["formula", "data", "limits", "conditions", "method", "other"]}},
                       "required": ["missing"]},
    },
}


# validated results

_SHOWN = ("exact", "numeric", "value", "root", "roots", "solution", "solutions", "objective", "parameters", "determinant",
          "eigenvalues", "dominant_frequencies", "final", "mean", "statistic", "p_value", "equivalent", "satisfied",
          "error_estimate", "residual")


def succeeded(call):
    return call.get("state") == "completed" and (call.get("reply") or {}).get("success") is True


def _compact(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return value if isinstance(value, str) else repr(value)
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 240 else text[:240] + "…"


def result_of(call):
    """A call that ran and gave a result, as the answer may use it."""
    if not succeeded(call):
        return None
    reply = call["reply"]
    return {
        "tool": call["name"],
        "operation": reply.get("operation") or "",
        "input": [line for line in reply.get("interpreted_input") or [] if isinstance(line, str)],
        "exact": reply.get("result_kind") == "exact",
        "result": [f"{key}: {_compact(reply[key])}" for key in _SHOWN if key in reply and reply[key] is not None],
        "reply": call.get("result") or "",
    }


def ledger(calls):
    return [r for r in (result_of(c) for c in calls if is_math(c.get("name"))) if r]


def _distinct(results):
    seen, out = set(), []
    for r in results:
        key = (r["tool"], r["operation"], tuple(r["result"]), tuple(r["input"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# reading numbers out of an answer

_SUPER = str.maketrans({"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
                        "⁻": "-", "⁺": "+"})
_NUMBER = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:\\times|\\cdot|×|·|\*)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"
    r"|10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"
    r"|(\d+(?:\.\d+)?)[eE]([-+]?\d+)"
    r"|\d+(?:[.,]\d+)?")
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*+>][ \t]*)?(?:#{1,6}[ \t]*)?(?:\*\*|__)?\(?\d{1,2}[.)](?=\s|\*|_)")


def _plain(text):
    text = str(text or "").replace("−", "-")
    text = re.sub(r"\\[,;:! ]|~", " ", text)
    out, raised = [], False
    for ch in text:
        mapped = ch.translate(_SUPER)
        if mapped != ch:
            if not raised:
                out.append("^")
            out.append(mapped)
            raised = True
        else:
            out.append(ch)
            raised = False
    return "".join(out)


def _significant(text):
    digits = "".join(c for c in text if c.isdigit()).lstrip("0")
    return max(1, len(digits))


def _decimal(text):
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def numbers(text):
    """(text, value, significant digits, is a bare power of ten) for every number in the text."""
    found = []
    for m in _NUMBER.finditer(_plain(text)):
        whole = m.group(0)
        if m.group(1) is not None and m.group(2) is not None:
            mantissa = _decimal(m.group(1))
            if mantissa is not None:
                found.append((whole, mantissa * 10 ** float(m.group(2)), _significant(m.group(1)), False))
        elif m.group(3) is not None:
            found.append((whole, 10 ** float(m.group(3)), 1, True))
        elif m.group(4) is not None:
            found.append((whole, float(m.group(4)) * 10 ** float(m.group(5)), _significant(m.group(4)), False))
        else:
            value = _decimal(whole)
            if value is not None:
                found.append((whole, value, _significant(whole), False))
    return found


def grounded(literal, values):
    """A number a source states, or that number rounded or cut to fewer digits. A short integer such
    as the 6 of 6ζ(4) is not taken for a rounding: it has to be stated as it is."""
    text, x, digits, power = literal
    x = abs(x)
    rounds = digits >= 3 or any(c in text for c in ".,eE^×*")
    for value in values:
        value = abs(value)
        if not math.isfinite(value):
            continue
        if x == value:
            return True
        if not rounds or x <= 0 or value <= 0:
            continue
        exponent = math.floor(math.log10(value))
        if power:
            k = round(math.log10(x))
            if k in (exponent, exponent + 1):
                return True
            continue
        unit = 10 ** (exponent - digits + 1)
        for candidate in (round(value / unit) * unit, math.trunc(value / unit) * unit):
            if abs(x - candidate) <= unit * 1e-3:
                return True
    return False


_CONSTANTS = [
    ("π", re.compile(r"π|\\pi(?![A-Za-z])|(?<![A-Za-z])pi(?![A-Za-z])")),
    ("ζ", re.compile(r"ζ|\\zeta(?![A-Za-z])|(?<![A-Za-z])zeta\s*\(")),
    ("Γ", re.compile(r"Γ|\\Gamma(?![A-Za-z])|(?<![A-Za-z])[Gg]amma\s*\(")),
    ("γ", re.compile(r"γ|\\gamma(?![A-Za-z])|EulerGamma")),
    ("Catalan", re.compile(r"(?<![A-Za-z])Catalan(?![A-Za-z])")),
    ("Li", re.compile(r"\\operatorname\{Li\}|(?<![A-Za-z])Li_|polylog\s*\(")),
    ("erf", re.compile(r"(?<![A-Za-z])erfc?\s*\(|\\operatorname\{erfc?\}")),
]


def constants(text):
    return [name for name, pattern in _CONSTANTS if pattern.search(str(text or ""))]


def ungrounded(answer, sources, results):
    """Numbers and named constants of an answer that neither the user's messages nor a validated result state."""
    known = list(sources) + [r["reply"] for r in results]
    values = [n[1] for text in known for n in numbers(text)]
    missing = []
    for literal in numbers(_LIST_MARKER.sub(" ", str(answer or ""))):
        if not grounded(literal, values) and literal[0] not in missing:
            missing.append(literal[0])
    named = {c for text in known for c in constants(text)}
    for name in constants(answer):
        if name not in named and name not in missing:
            missing.append(name)
    return missing


# what becomes of a round that ended without tool calls

STANDING_NOTE = ("Answer in prose with the values the tools returned, written as they returned them; do not paste the "
                 "raw tool output. Do not add digits, constants, closed forms or intermediate steps that no tool "
                 "returned; say instead what was not computed.")


def _listing(results):
    if not results:
        return "none"
    return "\n".join(
        f"- {r['tool']} {r['operation']} ({'exact' if r['exact'] else 'approximate'}): " + "; ".join(r["result"])
        + (". Input: " + "; ".join(r["input"]) if r["input"] else "")
        for r in _distinct(results))


def final_note(results):
    return ("Some math calls of this turn were refused or gave no result, and no more tools can be called. Write the "
            "final answer now from the validated results below: state them, then name in words, without any formula "
            "or number, the parts of the request that could not be validated with the tools. Do not work those parts "
            "out yourself, and do not state any number, constant or formula that is not in the request or in these "
            "results.\n\nValidated results:\n" + _listing(results))


def retry_note(missing, results):
    return ("Your answer states values that are not in the request or in a validated tool result: "
            + ", ".join(missing[:8]) + ". Leave them out and say they could not be validated; only if the request "
            "itself gives everything a calculation needs, compute it with a math tool instead. Never work them out "
            "yourself. Then give the answer again.\n\nValidated results:\n" + _listing(results))


def is_safe_answer(answer):
    return any(str(answer or "").startswith(t) for t in _TEXTS["header"].values())


def safe_answer(results, calls, lang="en"):
    """The answer Tosh writes itself when the model's own cannot be shown."""
    if not results:
        return unresolved(lang=lang)
    lines = [text("header", lang), ""]
    for r in _distinct(results):
        title = r["operation"][:1].upper() + r["operation"][1:]
        lines.append(f"- **{title}** ({text('exact' if r['exact'] else 'approximate', lang)}): " + " · ".join(r["result"]))
        if r["input"]:
            lines.append("  ▸ " + " · ".join(r["input"]))
    lines.append("")
    if not any(r["exact"] for r in results):
        lines.append(text("no_exact", lang))
    replies = [c.get("reply") or {} for c in calls]
    opened = []
    for reply in replies:
        if error_code(reply) == "no_closed_form" and isinstance(reply.get("unevaluated"), str):
            if reply["unevaluated"] not in opened:
                opened.append(reply["unevaluated"])
    for expression in opened:
        lines.append(text("open", lang).format(expression))
    if any(error_code(r) in REJECTIONS for r in replies):
        lines.append(text("refused", lang))
    lines.append(text("rest", lang))
    return "\n".join(lines)


def step(sources, calls, answer, closing, finalizing, regrounded, required, lang="en"):
    """keep, ("replace", text), ("again", note) or ("finalize", note) for a round without tool calls."""
    math_calls = [c for c in calls if is_math(c.get("name"))]
    if not math_calls and not required:
        return "keep", None
    results = ledger(math_calls)
    if closing is not None:
        return ("keep", None) if not results or finalizing else ("finalize", final_note(results))
    if finalizing and not str(answer or "").strip():
        return "replace", safe_answer(results, math_calls, lang)
    missing = ungrounded(answer, sources, results)
    if not missing:
        return "keep", None
    if finalizing or regrounded:
        return "replace", safe_answer(results, math_calls, lang)
    return "again", retry_note(missing, results)
