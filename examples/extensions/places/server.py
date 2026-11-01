# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""MCP server over stdio, standard library only: well-known places, the person's own (a JSON setting, GeoJSON or
CSV files) and distances between them. The `about` tool reports the settings it sees."""

import csv
import json
import math
import os
import sys
import unicodedata

# name, category, country, latitude, longitude, elevation in metres (mountains), other names
BUILT_IN = [
    ("Madrid", "capital", "Spain", 40.4168, -3.7038, None, []),
    ("Caracas", "capital", "Venezuela", 10.4806, -66.9036, None, []),
    ("Paris", "capital", "France", 48.8566, 2.3522, None, ["París"]),
    ("London", "capital", "United Kingdom", 51.5074, -0.1278, None, ["Londres"]),
    ("Rome", "capital", "Italy", 41.9028, 12.4964, None, ["Roma"]),
    ("Berlin", "capital", "Germany", 52.5200, 13.4050, None, ["Berlín"]),
    ("Kyiv", "capital", "Ukraine", 50.4501, 30.5234, None, ["Kiev", "Київ"]),
    ("Tokyo", "capital", "Japan", 35.6762, 139.6503, None, ["Tokio"]),
    ("Beijing", "capital", "China", 39.9042, 116.4074, None, ["Pekín"]),
    ("New Delhi", "capital", "India", 28.6139, 77.2090, None, ["Nueva Delhi"]),
    ("Cairo", "capital", "Egypt", 30.0444, 31.2357, None, ["El Cairo"]),
    ("Nairobi", "capital", "Kenya", -1.2921, 36.8219, None, []),
    ("Canberra", "capital", "Australia", -35.2809, 149.1300, None, []),
    ("Washington, D.C.", "capital", "United States", 38.9072, -77.0369, None, ["Washington"]),
    ("Ottawa", "capital", "Canada", 45.4215, -75.6972, None, []),
    ("Mexico City", "capital", "Mexico", 19.4326, -99.1332, None, ["Ciudad de México"]),
    ("Bogotá", "capital", "Colombia", 4.7110, -74.0721, None, []),
    ("Lima", "capital", "Peru", -12.0464, -77.0428, None, []),
    ("Brasília", "capital", "Brazil", -15.7939, -47.8828, None, []),
    ("Buenos Aires", "capital", "Argentina", -34.6037, -58.3816, None, []),
    ("Mount Everest", "mountain", "Nepal and China", 27.9881, 86.9250, 8849, ["Everest", "Monte Everest"]),
    ("Aconcagua", "mountain", "Argentina", -32.6532, -70.0109, 6961, []),
    ("Denali", "mountain", "United States", 63.0692, -151.0070, 6190, []),
    ("Kilimanjaro", "mountain", "Tanzania", -3.0674, 37.3556, 5895, []),
    ("Pico Bolívar", "mountain", "Venezuela", 8.5414, -71.0461, 4978, []),
    ("Mont Blanc", "mountain", "France and Italy", 45.8326, 6.8652, 4806, ["Montblanc"]),
    ("Matterhorn", "mountain", "Switzerland and Italy", 45.9763, 7.6586, 4478, ["Cervino"]),
    ("Mount Fuji", "mountain", "Japan", 35.3606, 138.7274, 3776, ["Fuji", "Monte Fuji"]),
    ("Teide", "mountain", "Spain", 28.2724, -16.6425, 3715, []),
    ("Eiffel Tower", "landmark", "France", 48.8584, 2.2945, None, ["Torre Eiffel"]),
    ("Colosseum", "landmark", "Italy", 41.8902, 12.4922, None, ["Coliseo"]),
    ("Sagrada Família", "landmark", "Spain", 41.4036, 2.1744, None, []),
    ("Statue of Liberty", "landmark", "United States", 40.6892, -74.0445, None, ["Estatua de la Libertad"]),
    ("Great Pyramid of Giza", "landmark", "Egypt", 29.9792, 31.1342, None, ["Gran Pirámide de Guiza"]),
    ("Taj Mahal", "landmark", "India", 27.1751, 78.0421, None, []),
    ("Machu Picchu", "landmark", "Peru", -13.1631, -72.5450, None, []),
    ("Angel Falls", "landmark", "Venezuela", 5.9701, -62.5362, None, ["Salto Ángel", "Kerepakupai Merú"]),
    ("Christ the Redeemer", "landmark", "Brazil", -22.9519, -43.2105, None, ["Cristo Redentor"]),
    ("Sydney Opera House", "landmark", "Australia", -33.8568, 151.2153, None, ["Ópera de Sídney"]),
]
CATEGORIES = ("capital", "mountain", "landmark", "custom")
EARTH_RADIUS_KM = 6371.0088
KM_PER_MILE = 1.609344
MAX_FILE_BYTES = 20 * 1024 * 1024

TOOLS = [
    {
        "name": "find",
        "description": "Looks up places: well-known capitals, mountains and landmarks, and the person's own. "
                       "Gives each place's coordinates, country and category.",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Part of a place's name, in English or Spanish"},
            "names": {"type": "array", "items": {"type": "string"}, "description": "Exact names to fetch"},
            "text": {"type": "string", "description": "A sentence: gives the places it names"},
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "country": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        }},
    },
    {
        "name": "near",
        "description": "Places within a distance of a place or of a point, nearest first. Distances in the "
                       "configured unit.",
        "inputSchema": {"type": "object", "properties": {
            "place": {"type": "string", "description": "A place's name; without it, the home place"},
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
            "radius": {"type": "number", "description": "In the configured unit; without it, the nearest ones"},
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        }},
    },
    {
        "name": "distance",
        "description": "Great-circle distance between two places, in the configured unit.",
        "inputSchema": {"type": "object", "properties": {
            "from": {"type": "string", "description": "A place's name; without it, the home place"},
            "to": {"type": "string"},
        }, "required": ["to"]},
    },
    {
        "name": "about",
        "description": "Which places and settings this extension has: counts by source, unit and home place.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def fold(text):
    """Lowercase without accents, so that "Bogota" finds Bogotá."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def place(name, category, country, latitude, longitude, elevation=None, aliases=(), source="built-in"):
    return {"name": name, "category": category, "country": country, "latitude": latitude, "longitude": longitude,
            "elevation_m": elevation, "aliases": list(aliases), "source": source}


def coordinate(value, limit):
    number = float(value)
    if not math.isfinite(number) or abs(number) > limit:
        raise ValueError(value)
    return number


def extra_places(problems):
    """TOSH_PLACES_EXTRA: a JSON object of names to [latitude, longitude] or {"latitude", "longitude", ...}."""
    text = os.environ.get("TOSH_PLACES_EXTRA", "").strip()
    if not text:
        return []
    try:
        entries = json.loads(text)
    except ValueError:
        problems.append("Extra places is not valid JSON")
        return []
    found = []
    for name, value in (entries.items() if isinstance(entries, dict) else []):
        try:
            if isinstance(value, list) and len(value) >= 2:
                found.append(place(name, "custom", "", coordinate(value[0], 90), coordinate(value[1], 180),
                                   source="extra"))
            elif isinstance(value, dict):
                found.append(place(name, "custom", str(value.get("country") or ""),
                                   coordinate(value.get("latitude"), 90), coordinate(value.get("longitude"), 180),
                                   value.get("elevation_m"), source="extra"))
        except (TypeError, ValueError):
            problems.append(f"Extra places: {name} has no valid coordinates")
    return found


def geojson_places(path, label):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    features = data.get("features", []) if isinstance(data, dict) else []
    found = []
    for feature in features:
        geometry = (feature or {}).get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        props = feature.get("properties") or {}
        name = props.get("name") or props.get("title") or props.get("NAME")
        longitude, latitude = geometry.get("coordinates", [None, None])[:2]
        if name:
            found.append(place(str(name), "custom", str(props.get("country") or ""), coordinate(latitude, 90),
                               coordinate(longitude, 180), source=label))
    return found


def csv_places(path, label):
    found = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row = {fold(k): v for k, v in row.items() if k}
            name = row.get("name") or row.get("nombre")
            latitude = row.get("latitude") or row.get("lat") or row.get("latitud")
            longitude = row.get("longitude") or row.get("lon") or row.get("lng") or row.get("longitud")
            if name and latitude and longitude:
                found.append(place(name, "custom", row.get("country") or "", coordinate(latitude, 90),
                                   coordinate(longitude, 180), source=label))
    return found


def file_places(problems):
    """TOSH_PLACES_FILES: the app's paths setting, a JSON object of short names to files or folders."""
    text = os.environ.get("TOSH_PLACES_FILES", "").strip()
    if not text:
        return []
    try:
        entries = json.loads(text)
    except ValueError:
        problems.append("Files is not valid JSON")
        return []
    found = []
    for label, path in (entries.items() if isinstance(entries, dict) else []):
        path = os.path.expanduser(str(path))
        files = [path]
        if os.path.isdir(path):
            files = sorted(os.path.join(path, n) for n in os.listdir(path) if not n.startswith("."))
        for file in files:
            kind = os.path.splitext(file)[1].lower()
            if kind not in (".geojson", ".json", ".csv") or not os.path.isfile(file):
                continue
            if os.path.getsize(file) > MAX_FILE_BYTES:
                problems.append(f"{os.path.basename(file)} is larger than 20 MB")
                continue
            try:
                read = csv_places if kind == ".csv" else geojson_places
                found.extend(read(file, label))
            except (OSError, ValueError, TypeError, AttributeError) as error:
                problems.append(f"{os.path.basename(file)} could not be read: {type(error).__name__}")
    return found


def settings():
    unit = "mi" if os.environ.get("TOSH_PLACES_UNITS") == "mi" else "km"
    # the app always sends a choices setting, empty when none is ticked; outside the app, all of them
    chosen = os.environ.get("TOSH_PLACES_CATEGORIES")
    categories = set(CATEGORIES) if chosen is None else {c.strip() for c in chosen.split(",") if c.strip()}
    return unit, categories, os.environ.get("TOSH_PLACES_HOME", "").strip()


def catalogue():
    """Read on every call, so a file the person edits counts at once."""
    problems = []
    places = [place(*entry) for entry in BUILT_IN] + extra_places(problems) + file_places(problems)
    _, categories, _ = settings()
    return [p for p in places if p["category"] in categories], problems


def lookup(places, name):
    wanted = fold(name)
    for candidate in places:
        if wanted in [fold(candidate["name"])] + [fold(a) for a in candidate["aliases"]]:
            return candidate
    partial = [p for p in places if wanted and wanted in fold(p["name"])]
    return partial[0] if len(partial) == 1 else None


def kilometres(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def in_unit(km, unit):
    return round(km / KM_PER_MILE if unit == "mi" else km, 1)


def public(p, **extra):
    shown = {k: v for k, v in p.items() if k not in ("aliases", "source") and v not in (None, "")}
    return dict(shown, **extra)


def text_reply(value, error=False):
    reply = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
    if error:
        reply["isError"] = True
    return reply


def find(arguments):
    places, _ = catalogue()
    limit = int(arguments.get("limit") or 20)
    if arguments.get("names"):
        hits = [p for p in (lookup(places, n) for n in arguments["names"] if isinstance(n, str)) if p]
    elif arguments.get("text"):
        sentence = " " + "".join(c if c.isalnum() else " " for c in fold(arguments["text"])) + " "
        hits = [p for p in places
                if any(" " + fold(n).replace(",", "").replace(".", " ").strip() + " " in sentence
                       for n in [p["name"]] + p["aliases"])]
    else:
        query = fold(arguments.get("query") or "")
        hits = [p for p in places if query in fold(p["name"]) or any(query in fold(a) for a in p["aliases"])]
    if arguments.get("category"):
        hits = [p for p in hits if p["category"] == arguments["category"]]
    if arguments.get("country"):
        hits = [p for p in hits if fold(arguments["country"]) in fold(p["country"])]
    return text_reply({"count": len(hits), "places": [public(p) for p in hits[:limit]]})


def origin(places, arguments, key):
    name = arguments.get(key) or settings()[2]
    if not name:
        return None, "no place given and no home place set"
    found = lookup(places, name)
    return (found, None) if found else (None, f"unknown place: {name}")


def near(arguments):
    places, _ = catalogue()
    unit = settings()[0]
    if arguments.get("latitude") is not None and arguments.get("longitude") is not None:
        try:
            centre = {"name": "point", "latitude": coordinate(arguments["latitude"], 90),
                      "longitude": coordinate(arguments["longitude"], 180)}
        except (TypeError, ValueError):
            return text_reply({"error": "invalid coordinates"}, error=True)
    else:
        centre, problem = origin(places, arguments, "place")
        if problem:
            return text_reply({"error": problem}, error=True)
    ranked = sorted(((kilometres(centre, p), p) for p in places if p is not centre), key=lambda pair: pair[0])
    if arguments.get("category"):
        ranked = [(d, p) for d, p in ranked if p["category"] == arguments["category"]]
    if arguments.get("radius") is not None:
        ranked = [(d, p) for d, p in ranked if in_unit(d, unit) <= float(arguments["radius"])]
    limit = int(arguments.get("limit") or 10)
    return text_reply({"from": centre["name"], "unit": unit,
                       "places": [public(p, distance=in_unit(d, unit)) for d, p in ranked[:limit]]})


def distance(arguments):
    places, _ = catalogue()
    start, problem = origin(places, arguments, "from")
    end = lookup(places, arguments.get("to") or "")
    if problem or end is None:
        return text_reply({"error": problem or f"unknown place: {arguments.get('to')}"}, error=True)
    unit = settings()[0]
    return text_reply({"from": public(start), "to": public(end), "distance": in_unit(kilometres(start, end), unit),
                       "unit": unit})


def about(_):
    places, problems = catalogue()
    unit, categories, home = settings()
    counts = {}
    for p in places:
        counts[p["source"]] = counts.get(p["source"], 0) + 1
    return text_reply({"places": counts, "unit": unit, "categories": sorted(categories), "home": home or None,
                       # a secret setting: a real extension would send it to its service, never show it
                       "service_key": "set" if os.environ.get("TOSH_PLACES_SERVICE_KEY") else "not set",
                       "problems": problems})


HANDLERS = {"find": find, "near": near, "distance": distance, "about": about}


def answer(message):
    method = message.get("method")
    if method == "initialize":
        version = (message.get("params") or {}).get("protocolVersion", "2025-06-18")
        return {"protocolVersion": version, "capabilities": {"tools": {}},
                "serverInfo": {"name": "places", "version": "1.0.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        params = message.get("params") or {}
        handler = HANDLERS.get(params.get("name"))
        if handler:
            try:
                return handler(params.get("arguments") or {})
            except (TypeError, ValueError) as error:
                return text_reply({"error": f"invalid arguments: {error}"}, error=True)
    if method == "ping":
        return {}
    raise LookupError(method)


def main():
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict) or "id" not in message:
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": message["id"], "result": answer(message)}
        except LookupError:
            reply = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "unknown method"}}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
