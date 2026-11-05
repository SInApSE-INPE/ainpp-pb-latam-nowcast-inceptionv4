# Decision records

Why the structure is the way it is. Each record states the context, the
decision, its consequences, and the alternatives rejected, so that a later
reader can tell a deliberate choice from an accident.

Read them in order if you want the reasoning behind the architecture; read
0001 and 0003 if you only want the two that shape everything else.

```{toctree}
:maxdepth: 1

adr/0001-pipeline-shape-scope
adr/0002-workspace-and-model-isolation
adr/0003-ports-and-their-granularity
adr/0004-optional-stages-and-built-in-adapters
adr/0005-conformance-as-the-contract
adr/0006-scaffolding-and-a-framework-free-runtime
```

## A note on corrections

Two of these records have been corrected in place rather than rewritten, and
both say so where the correction applies. A decision record that quietly
becomes false is worse than no record at all, because a reader has no way to
tell. When a later decision invalidates an earlier consequence, the earlier
record is marked and points forward.
