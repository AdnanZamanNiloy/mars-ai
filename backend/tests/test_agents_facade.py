"""Guards the lazy `app.agents` public surface against drift.

The facade once advertised modules that never existed (`mission`, `trace`,
`decision`, `evaluation`), so `getattr(app.agents, "run_mission")` raised
ModuleNotFoundError for any consumer that trusted the docstring. This test
fails the moment an entry points at a missing module or a missing symbol,
which is the cheapest possible catch for that class of bug (AGENTS.md 4.5).
"""
import app.agents as agents


def test_contract_types_are_not_reexported():
    """`mission`/`trace`/`decision`/`evaluation` modules do not exist.

    Their names must not silently reappear in the facade without the module
    being built in the same commit — a dangling export is the exact trap
    this suite exists to prevent.
    """
    phantom_modules = {"app.agents.mission", "app.agents.trace",
                       "app.agents.decision", "app.agents.evaluation"}
    referenced = set(agents._LAZY.values())
    assert not (referenced & phantom_modules)


def test_every_lazy_export_resolves():
    """Every promised name imports from the module it claims to live in."""
    broken = []
    for name, module_path in agents._LAZY.items():
        try:
            getattr(agents, name)
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            broken.append(f"{name} -> {module_path}: {type(exc).__name__}: {exc}")
    assert not broken, "Broken facade exports:\n" + "\n".join(broken)


def test_dir_lists_promised_names():
    """`dir()` advertises exactly the stable surface callers can rely on."""
    advertised = set(dir(agents))
    assert set(agents._LAZY).issubset(advertised)
