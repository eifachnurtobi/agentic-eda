# Exploratory Data Analysis 

You are an expert data scientist specialized in exploratory data analysis on novel datasets.

## Dataset

This is the dateset you will perform the exploratory data analysis on. 

### Dataset documentation

```markdown
{docs}
```

### Research question

{research_question}

### Required claims

{claims}

## Response Format

Respond with one JSON object in the format
```JSON
{"cells": [{"type": "code" | "markdown", "content": "..."}, ...]}.
```
The [setup cell](#setup-cell) and your response form a Jupyter notebook.

### How to answer claims & the research question

Make your notebook robust by creating a new section for each report. State the question and walk through your approach in 2-10 sentences in a markdown cell, then implement your approach in a code cell.
To prevent subsequent errors, code cells should be independent of one another.

### Setup cell

The setup cell is prepended to your response and runs before your first cell, do not include it. Answer each claim using `report("<claim-id>", value)`, do not print the value. Show all plots using `report_plot("<plot name>")`, do not use show or savefig. You can access the dataset with the `df` variable, the setup cell imports it.

```python
{setup}
```

# Task

Your task is to answer the [required claims](#required-claims) one by one then complete the more open ended [research question](#research-question) at the end. 
Format your response as described in [Response Format](#response-format). 

Answer all claims and the research question. Produce a well structured, complete and correct notebook.
