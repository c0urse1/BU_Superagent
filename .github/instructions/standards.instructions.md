---
applyTo: '**'
---
Please apply Strict Architecture Mode (SAM) when generating code or refactors.
Always structure proposals by layer: domain → application → infrastructure → interface → config.
- Domain: pure functions/classes, no I/O, no globals, no environment access.
- Application: use-cases + ports; orchestrates domain; no direct infra.
- Infrastructure: adapters implementing ports; wrap external libs (Chroma, OpenAI, DB, FS).
- Interface: CLI/HTTP; only parse/format + delegate to use-cases.
- Errors: typed classes or Result objects, never silent fails.
- Config: only in composition root.
Prefer correctness & architecture over speed or pragmatism.