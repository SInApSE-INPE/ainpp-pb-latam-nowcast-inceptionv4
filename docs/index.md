# Nowcast Platform

Operational infrastructure for publishing gridded nowcasts on a schedule, with
auditable provenance, and the models that run on it.

```{admonition} What this is for
:class: tip

You have a trained forecasting model. You want it producing files on a schedule,
reproducibly, with a record of exactly what produced each one. This platform
supplies everything around the model so that you supply only the model.
```

The scope is deliberately narrow:

> Gridded geospatial sequence-to-sequence forecasting, with scheduled batch
> publication and auditable provenance.

Anything outside that is refused rather than accommodated. See
[Scope and limits](about/scope.md) for what that admits and what it does not.

## Start here

| | |
| --- | --- |
| **[Install](getting-started/installation.md)** | Set up the workspace and run the checks. |
| **[Quickstart](getting-started/quickstart.md)** | Generate a working model and watch it publish, in about five minutes. |
| **[Your first model](tutorials/01-your-first-model.md)** | A full walkthrough: replace the placeholder with a real network. |
| **[How it works](concepts/architecture.md)** | The pipeline spine, the ports, and why there are exactly these. |

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/01-your-first-model
tutorials/02-a-new-data-source
tutorials/03-ensembles-and-variables
tutorials/04-running-in-a-container
tutorials/05-scheduling-and-operations
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/architecture
concepts/ports
concepts/grids-and-windows
concepts/stages
concepts/artifacts-and-manifests
```

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

guides/configuration
guides/conformance
guides/troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/cli
reference/api
```

```{toctree}
:maxdepth: 1
:caption: About
:hidden:

about/scope
decisions
```
