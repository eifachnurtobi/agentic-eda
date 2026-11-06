agentic-eda/
├── README.md              # Anleitung: Installation, Start, Befehle und Projektaufbau
├── pyproject.toml         # Python-Paket, Abhängigkeiten sowie Pytest- und Ruff-Konfiguration
├── uv.lock                # Exakt festgelegte Versionen aller Python-Abhängigkeiten
├── .python-version        # Vorgesehene Python-Version: 3.12
├── .gitignore             # Dateien, die Git ignoriert, z. B. runs/ und .venv/
├── compose.yaml           # Sicherheits- und Ressourcenregeln der Docker-Sandbox
│
├── docker/
│   ├── Dockerfile         # Baut das Sandbox-Image
│   ├── requirements.lock  # Python-Pakete innerhalb der Sandbox
│   └── nb_executor.py     # Führt generierte Notebooks Zelle für Zelle aus
│
├── src/eda_lite/
│   ├── __init__.py        # Markiert eda_lite als Python-Paket
│   ├── cli.py             # Kommandozeile: run, prompt, trace und compare
│   ├── config.py          # Lädt und validiert project.yaml
│   ├── grade.py           # Vergleicht Ergebnisse mit answers.yaml und berechnet den Score
│   ├── model.py           # Kommunikation mit Ollama und Verarbeitung der Modellantwort
│   ├── patterns.py        # Agentenstrategien wie Zero-Shot und Reflection
│   ├── prompts.py         # Lädt und befüllt die Prompt-Templates
│   ├── report.py          # Erstellt Zusammenfassungen, Traces und Laufvergleiche
│   ├── run.py             # Zentrale Ablaufsteuerung, Budgets und Logging
│   ├── sandbox.py         # Startet die sichere Notebook-Ausführung in Docker
│   └── prompts/
│       ├── zero_shot.md   # Hauptprompt für einen direkten Modellaufruf
│       ├── reflection.md  # Prompt zur Kritik und Verbesserung einer ersten Antwort
│       └── setup_cell.py  # Vertrauenswürdige erste Notebook-Zelle mit df/report-Funktionen
│
├── projects/
│   ├── stix_flares_simple/    # Einfache Aufgabe mit wenigen, direkten Claims
│   ├── stix_flares/           # Mittlere Aufgabe zur zeitlichen Entwicklung der Flares
│   └── stix_flares_extensive/ # Umfangreiche Aufgabe mit vielen komplexen Claims
│
└── tests/
    ├── conftest.py            # Gemeinsame Testdaten und Hilfsfunktionen
    ├── test_lite.py           # Unit- und Docker-End-to-End-Tests
    └── fixtures/tiny_project/ # Kleines Testprojekt mit CSV, Config und Musterantworten