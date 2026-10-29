# Contributing

Work in the module assigned in README.md. Agree on shared contract changes before integrating dependent modules. Use pull requests and describe the behavior, validation, and limitations. Keep secrets and generated datasets out of Git.

Implement modules against `shared/contracts.py`. Use stable paper IDs, retain provider IDs and evidence source locations, and report unavailable text or API failures explicitly. Do not replace unavailable evidence with generated text. Keep network-dependent checks separate from deterministic tests.
