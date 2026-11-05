# 0006 — Scaffold something that already works, and keep the runtime framework free

Status: Accepted
Date: 2026-11-05

## Context

The requirement driving this platform is that an adopter's effort be
proportional to what is genuinely unique about their model. The contracts and
the conformance suite make that checkable, but they do not make it *easy*: a
newcomer still has to assemble a plugin, a composition root, a configuration, a
bundle and a test before anything runs at all, and until then every mistake
looks the same.

## Decision

### Generate a project that passes conformance before anything is written

`nowcast-new-model` writes a complete project: a plugin, a composition root, a
configuration, a placeholder bundle with a real SHA-256 manifest, a Dockerfile,
and a conformance test already wired to the shared suite. It ships a persistence
predictor, so `uv run pytest` is green immediately.

The adopter then replaces one thing at a time and watches what breaks. Every
spot that must change carries a `REPLACE THIS` marker, and the generated README
lists them in the order they should be done.

Starting from something that runs, rather than from something that must be
assembled before it can run, is the whole point. A failure then means *the thing
I just changed*, which is the only diagnostic that scales to someone who has not
read this repository.

Templates substitute with `$name`, not `{name}`, so generated Python can contain
braces — f-strings, dict literals, regex quantifiers — without escaping.

A test generates a project and runs its suite in a subprocess. The promise is
mechanically checked rather than asserted in prose.

### The runtime depends on no deep learning framework

ADR 0003 established that no port mentions a tensor type, so a model on another
framework needs no new contract. Generating a NumPy-only persistence model
revealed that this was true of the *contracts* and false of the *dependencies*:
the runtime still declared `torch`, left over from before the pipeline was
migrated onto the ports, and nothing imported it any more.

`torch` is removed. A model brings its own framework, or none.

Two tests hold the invariant: one reads the runtime's declared dependencies and
one greps its modules, because this is exactly the kind of property that gets
reintroduced by a convenient import.

## Consequences

- A generated project installs in seconds rather than downloading 200 MB it
  never uses, and a model on TensorFlow, JAX or plain NumPy is no longer made to
  carry torch.
- The workspace lock lost torch, sympy and their transitive weight, so the
  runtime's own checks run on a much smaller environment.
- A coupling documented in [ADR 0002](0002-workspace-and-model-isolation.md)
  dissolved. The container build context needed the workspace root only because
  that file declared the torch index the runtime resolved through. With no torch
  in the runtime, the lock is reproducible without it, and both Dockerfiles have
  dropped the copy. That record is corrected in place rather than left to
  mislead.
- The scaffold is now a surface that must be maintained: a change to a port is a
  change to a template, and the subprocess test is what makes that failure loud.

## Alternatives rejected

- **Documentation instead of generation.** Cheaper, and it puts the whole
  assembly burden on the reader at exactly the moment they know least. The
  existing documentation already carried statements that silently went stale,
  which is the argument against relying on more of it.
- **An empty skeleton with `NotImplementedError` stubs.** Honest about what is
  missing, and it cannot be run, so the first feedback an adopter gets is a wall
  of failures with no way to tell which are expected.
- **Keeping `torch` as a convenience.** It is one line and it costs every
  adopter a large download for something the runtime does not use. Worse, it
  quietly contradicts the claim that the platform is framework neutral, which is
  the kind of gap that makes the rest of the documentation less trustworthy.
- **A cookiecutter dependency.** A well-trodden tool, and a new dependency plus
  a new file format to express what `string.Template` already does for a
  generator this size.
