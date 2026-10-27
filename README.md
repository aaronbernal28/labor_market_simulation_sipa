# labor_market_simulation_sipa

This repository contains a simulation of the labor market based on the occupational relations described in *"Redes ocupacionales: estructura del mercado laboral y brecha de género en Argentina"* and microdata from the *Muestra Longitudinal de Empleo Registrado* (MLER / SIPA) dataset.

## Bibliography

```latex
@article{Carrascosa_Bernal Huanca_Semeshenko_Sarraute_2026,
	title={Redes ocupacionales: estructura del mercado laboral y brecha de género en Argentina},
	volume={12},
	url={https://revistas.unlp.edu.ar/JAIIO/article/view/22171},
	abstractNote={Este trabajo analiza la estructura del mercado laboral argentino mediante un enfoque de redes complejas que vincula ocupaciones (CIUO-08) y ramas de actividad económica (CAES 1.0). Utilizando microdatos de las encuestas ENES (2019) y ESAyPP (2021), se construye una red bipartita de alta resolución y sus correspondientes proyecciones unipartitas para desentrañar patrones de interdependencia no visibles con métodos tradicionales. Los resultados revelan una organización modular donde los sectores industriales y los servicios sociales ocupan regiones diferenciadas de la red, mientras que el comercio actúa como un nodo puente que articula ambos espacios. En la red ocupacional se observa una marcada segmentación entre ocupaciones manuales y no manuales, junto con la presencia de ocupaciones transversales que articulan múltiples sectores económicos. Un hallazgo central es que la segregación de género posee un correlato topológico: las ocupaciones con composiciones de género similares tienden a agruparse en la red (homofilia), evidenciando que la desigualdad está escrita en la propia estructura relacional del mercado. Estos resultados muestran el potencial del análisis de redes para comprender la organización estructural del mercado laboral y sus patrones de segmentación.},
	number={1},
	journal={JAIIO, Jornadas Argentinas de Informática},
	author={Carrascosa, Joaquín and Bernal Huanca, Aaron and Semeshenko, Viktoriya and Sarraute, Carlos},
	year={2026},
	month={Oct.},
	pages={60–74}
}
```

## Repository Structure

```text
labor_market_simulation_sipa/
├── Snakefile              ← Pipeline orchestration and top-level target rules
├── config.yaml            ← Central parameters, file paths, and schemas
├── pyproject.toml         ← Python library packaging and pinned dependencies
├── ruff.toml              ← Linter and formatter configuration (tab-indentation)
├── .env.example           ← Template for environment variables and secrets
├── workflow/
│   ├── rules/             ← Modular Snakemake rule definitions (*.smk)
│   ├── schemas/           ← JSON Schema validation for config.yaml
│   └── envs/              ← Conda environment specifications
├── scripts/               ← Thin Tier 2 adapter scripts invoked by rules
├── src/                   ← Pure modular Tier 1 domain library
│   ├── pipeline/          ← Data loaders, bipartite builders, and transition matrices
│   └── simulation/        ← Agent-based model (ABM) simulations
├── tests/                 ← Unit test suite for domain library components
├── data/                  ← Raw, partitioned Parquet, and transition artifacts
└── images/                ← Generated plots and diagnostics
```

## Environment Setup

Create and activate the virtual environment, install dependencies, and package the library in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start (Domain Library)

```python
import src as lms

loader = lms.MLERLoader("data/processed/mler_parquet")
df = loader.load_period(199612)
```

## Snakemake Execution

Execute dry-runs, run lint validation, or trigger the full pipeline DAG:

```bash
# Validate workflow DAG
snakemake -n

# Check rule and directive linting
snakemake --lint

# Run full pipeline with all available cores
snakemake --cores all
```
