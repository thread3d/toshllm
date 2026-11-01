# Extensions

An extension adds tools to ToshLLM's engine, and optionally its own agent, with a settings page in
the app. It is a folder with an `extension.json` and whatever programs that file starts. The servers
it declares are ordinary [MCP](https://modelcontextprotocol.io) servers, so anything that speaks MCP
can write one, and they also run without ToshLLM. [`examples/extensions/clock`](../examples/extensions/clock)
is a complete one in about seventy lines of standard-library Python, and
[`examples/extensions/places`](../examples/extensions/places) uses every kind of setting and brings an agent
that streams its answer, hands the turns outside its field back to the chat and offers a map.

## Where ToshLLM looks

- **Settings → Extensions → Extensions folder**: a folder of your choice, by default
  `~/Library/Application Support/ToshLLM/Extensions`. Every subfolder with an `extension.json` is
  one extension; a chosen folder that has an `extension.json` itself is a single extension.
- **Inside the app** (`Contents/Resources/extensions`), for builds that bring their own
  (`TOSH_APP_EXTENSIONS=<folder>` in `make-app.sh`). Such a build can also set its own defaults for
  ToshLLM's settings with `TOSH_APP_DEFAULTS=<plist>`, keyed like `SettingsKeys`; what a user changes still wins.

An extension found in the folder starts switched off. Switching it on shows the programs it will run;
they run with your user's permissions, so only approve extensions you trust. The approval belongs to
that exact `extension.json`: if the file changes, the extension switches off until approved again.
Extensions inside the app start on. Changes apply when the server restarts.

## extension.json

```json
{
  "format": 1,
  "id": "weather",
  "version": "1.2.0",
  "name": {"es": "Tiempo", "en": "Weather"},
  "description": {"es": "Previsiones de un servicio meteorológico.", "en": "Forecasts from a weather service."},
  "author": "Example Ltd.",
  "servers": {
    "weather": {
      "command": "{python}",
      "args": ["-I", "-B", "{extension}/server.py"],
      "timeout_ms": 30000,
      "enabled_by": "WEATHER_ENABLED"
    },
    "radar": {
      "url": "https://mcp.example.com/radar",
      "header_env": {"Authorization": "WEATHER_TOKEN"}
    }
  },
  "settings": [
    {
      "id": "weather",
      "title": {"es": "Tiempo", "en": "Weather"},
      "fields": [
        {"key": "WEATHER_ENABLED", "kind": "toggle", "default": "1",
         "title": {"es": "Activo", "en": "On"}, "help": {"es": "…", "en": "…"}},
        {"key": "WEATHER_TOKEN", "kind": "secret",
         "title": {"es": "Clave", "en": "Key"}, "help": {"es": "…", "en": "…"}}
      ]
    }
  ]
}
```

| Key | Meaning |
|---|---|
| `format` | `1` |
| `id` | Lowercase letters, digits and `-`; unique among the extensions in use. |
| `version`, `author` | Shown in Settings. |
| `name`, `description` | Text in Spanish and English (`{"es", "en"}`), like every text in the file. |
| `servers` | MCP servers the engine starts, by name (lowercase letters, digits, `-` and `_`; not `sympy`, `scientific` or `agent`, which ToshLLM uses). Their tools reach the model as `<name>_<tool>`. |
| `agent` | Optional. A local server that replaces the engine's agent; see below. |
| `math_tools` | Optional, `true` to start ToshLLM's symbolic and scientific math servers (`sympy`, `scientific`) with it even when they are switched off, for an agent that relies on them. |
| `settings` | Optional. Sections of the settings page. |

### Servers

A **local** server is a program the engine starts and talks to over stdio:

| Key | Meaning |
|---|---|
| `command`, `args` | The program and its arguments. |
| `env` | Extra environment for it. |
| `timeout_ms` | Longest a call may take (default 60000). |
| `enabled_by` | A `toggle` setting that switches the server; without one it always runs. |
| `agent_only` | Its tools are left out of the engine's `/tools`, so API clients and the chat's own tool loop never see them; the agent still does. |

The engine's agent, ToshLLM's own or the one an extension brings, offers every server's tools to the
model, so they work in the chat without further setup.

A **remote** server has a `url` instead of `command` and is reached over Streamable HTTP, or the
older HTTP+SSE transport (`"transport": "http"` or `"sse"` skips the detection). `header_env` maps
an HTTP header to the setting that holds its value. `timeout_ms`, `enabled_by` and `agent_only`
work the same.

In `command`, `args`, `env` and `url`:

| Placeholder | Becomes |
|---|---|
| `{extension}` | This extension's folder. |
| `{python}` | The Python that ships with ToshLLM (3.13, with the standard library, SymPy, NumPy and SciPy). |
| `{runtime}` | The folder of that Python and ToshLLM's own helpers. |
| `{trust_key}` | A key, new on every start, that only the app and the servers it starts know. |

A server is free to bring its own runtime (`node`, a binary in `{extension}`, a Python of its own).

### Settings

Each section has an `id`, a `title`, an optional `detail` and `fields`. An optional `notice`
(`{"when_empty": [keys], "text": {…}}`) is shown while all those settings are empty. A field has a
`key`, a `kind`, a `title` and a `help`, and optionally a `default` and a `placeholder`:

| Kind | Stored as |
|---|---|
| `toggle` | `1` or `0`; a switched-off toggle is left out of the environment. |
| `text` | The text as written. |
| `secret` | Kept in the macOS Keychain, never shown again. |
| `choice` | One of `options` (`[{"value", "title"}]`). |
| `choices` | Any of `options`, as their values joined by commas in the order of `options`; `default` is such a list. It always reaches the engine, empty when none is ticked. |
| `json` | A JSON object, saved once it parses. |
| `paths` | A JSON object of short names to files or folders the person picks (`file_types`, `folder_types`; `"folder_types": ["*"]` takes any folder). |

Every setting that holds a value reaches the engine as an environment variable named by its `key`,
and every server the engine starts inherits it, so a secret never shows on a command line. Use
names that are unlikely to clash with other extensions.

### The agent

`agent` is a local server, declared like the others, that the engine runs as `--mcp-agent`: every
agent turn of the chat, the web chat and API clients goes to its `run` tool, which answers it using
the model and the tools through the engine. The contract is described in
[`helpers/tosh-sympy/README.md`](../helpers/tosh-sympy/README.md#the-tosh-agent), and what another
agent has to send back, including how it hands a turn back to the chat's own tools and how it
offers a map of an answer, in [its next section](../helpers/tosh-sympy/README.md#another-agent). Only one extension with an agent
can be on at a time.

## Without ToshLLM

The servers are standard MCP servers over stdio or HTTP, so any MCP client can start them with the
same `command` and `args` once the placeholders are filled in and the settings are set as
environment variables. ToshLLM's engine takes them with
`llama-server --mcp-servers-json '{"mcpServers": {...}}'`, and the agent with `--mcp-agent <name>`.
