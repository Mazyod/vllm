# Engine configuration store

The store makes the configuration named by a result the same file vLLM read.
Each release directory has two layers:

```text
configs/<TAG>/
  fleet.yaml
  engine/*.yaml
  results/*.md
```

`engine/*.yaml` contains only arguments accepted by that release's `vllm
serve --config`. `fleet.yaml` contains the launch and probe metadata that vLLM
cannot consume. One engine file represents one distinct argument set and may be
shared by profiles that differ only in environment, phase, probes, GPU
assignment, or replica count.

## Engine schema

An engine file is a top-level YAML mapping. Keys are the long `vllm serve`
option names without `--`; values have the types accepted by that release's
parser. Nested argument values such as `speculative-config` and
`compilation-config` are YAML mappings rather than quoted JSON.

Gate engine files provide `model`, `served-model-name`, and
`tensor-parallel-size` because the profile loader derives their model identity
and topology from the file. Manual engine files need only the arguments the
engine itself requires; they are indexed under `manual` rather than loaded as
gate profiles.

Comments belong beside load-bearing values. They preserve why an explicit
setting exists when a later release changes its default or makes the workaround
look redundant.

## Fleet schema

`fleet.yaml` has a `profiles` mapping for configurations launched by the gate
and an optional `manual` mapping for configurations operated outside it.

Every `profiles.<id>` entry has the same fields:

| field | meaning |
| --- | --- |
| `engine` | release-relative path under `engine/` |
| `phase` | runbook phase that schedules the profile |
| `env` | non-engine environment passed to each replica |
| `gpus` | GPU indices available to the profile |
| `replicas` | number of servers that form the measured fleet |
| `revert_patches` | patch files removed for a leave-one-out arm |
| `probes` | receipt, behavioural, negative, or performance probes to run |
| `expect` | `serves` or `boot_crash` |
| `expect_boot_evidence` | boot evidence required for the arm to inform a verdict |
| `expect_attention_backend` | backend receipt expected from this model, or an empty string when none is asserted |
| `gating` | whether an unexpected failure blocks the release |
| `control_for` | shipping profile this same-box control compares with, or `null` |
| `venue` | `gate`; off-gate entries belong under `manual` |

The profile id is stable result identity. `model`, `served-model-name`,
`tensor-parallel-size`, and the external draft model are deliberately absent
from the fleet layer because they are derived from the engine file.

Every `manual.<id>` entry requires `engine` and `venue: manual`. Its remaining
fields record the validation venue, image, date, status, and other evidence
needed to interpret that off-gate configuration. Manual entries are validated
against the release parser but are never scheduled by the gate.

## Parity contract

The launcher passes exactly this engine argv:

```text
vllm serve --config <resolved-path> --host 0.0.0.0 --port <replica-port>
```

No model or engine option is synthesized on the command line. `host` and
`port` are the closed runtime override set: the host is fixed by the harness,
and each replica needs its own port. The keys `host`, `port`, and `config` are
therefore forbidden in every engine file. Local and container launchers resolve
the same committed bytes to paths visible in their respective environments.

The parity oracle in
[`../tests/fixtures/legacy_serve_argv.json`](../tests/fixtures/legacy_serve_argv.json)
freezes the argv measured before the cutover. The config-store tests expand the
YAML and compare it with that witness, then use the release's real parser when
it is importable. A parser skip cannot erase the pure-argv comparison.

## Loader traps

The vLLM YAML loader does not translate every top-level YAML value into an
argument:

- `false` emits nothing.
- `[]` emits nothing.
- An unknown top-level key with either value is therefore silently ignored before
  argparse can reject it.

Those values mean "use the engine default", not "turn this off". A disabled
BooleanOptionalAction must use its explicit negative key, for example:

```yaml
no-enable-prefix-caching: true
```

The structural lint rejects top-level `false`, empty lists, `null`, duplicate
keys, short aliases such as `tp`, dotted keys, nested `config`, and the banned
runtime keys. A Boolean inside a structured mapping remains part of that
mapping; it is not interpreted as a top-level flag. The lint also rejects
symlinks and engine paths that escape their release directory. The real-parser
check catches keys or values the named vLLM release does not accept. Historical
directories are validated with their own release, never a newer's parser.

## Adding or changing a configuration

1. Create `configs/<TAG>/engine/<name>.yaml` with the complete long-form engine
   argument set. Reuse an existing engine file when the arguments are byte-for-
   byte the same.
2. Add gate profiles to `profiles` or off-gate records to `manual` in that
   release's `fleet.yaml`. Do not duplicate values the profile loader derives
   from the engine file.
3. Add or update the independent parity witness before accepting an engine
   change. A passing comparison must still describe the invocation intended for
   that release.
4. On a box running the named release, validate every file with:

   ```bash
   python3 -m fork.bench.config_validation --tag <TAG>
   ```

   The command must print the installed release version. A version mismatch or
   parser rejection is a failed validation.
5. Update [CATALOG.md](CATALOG.md) and add the release result record when the
   configuration has measurements.
6. Run `bash fork/bench/preflight.sh` before provisioning.

Local harness commands that import the store run through `uv run --no-project`
with `--with pyyaml`. The on-box image already supplies the release parser and
its YAML dependency.

## Freeze rule

An engine file freezes as soon as any run records a launch against it. Its
SHA-256 is result identity, so changing the file would make the committed path
describe different bytes from the recorded launch. Put every later change in a
new release directory and leave the launched file untouched.
