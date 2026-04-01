# 3D Droplet Coalescence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a separate educational 3D application that simulates coalescence of two water droplets in zero gravity, provides an interactive interface, and supports a computational experiment.

**Architecture:** Reuse the existing reduced-order physics approach from the 2D app, but generate an axisymmetric 3D surface by revolving a meridional profile extracted from an implicit field. Use a separate Streamlit app with a dedicated core module, sweep utilities, and a headless self-test.

**Tech Stack:** Python, NumPy, Plotly, Streamlit, Matplotlib, Pytest

---

### Task 1: Scaffold the 3D project

**Files:**
- Create: `droplet_coalescence_3d/requirements.txt`
- Create: `droplet_coalescence_3d/README.md`
- Create: `droplet_coalescence_3d/run_web.command`

**Step 1: Write the failing test**

Create a smoke test placeholder that imports the upcoming core module and fails because the file does not exist yet.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: FAIL with import/module error.

**Step 3: Write minimal implementation**

Create the project files and baseline dependency list.

**Step 4: Run test to verify it still reaches the missing core error**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: FAIL only because core implementation is missing.

**Step 5: Commit**

Not applicable unless the workspace is later initialized as a git repository.

### Task 2: Implement the 3D physics core

**Files:**
- Create: `droplet_coalescence_3d/coalescence_core_3d.py`
- Create: `droplet_coalescence_3d/test_core_3d.py`

**Step 1: Write the failing test**

Add tests for:

- parameter validation;
- simulation array shapes and monotonic time;
- finite `merge_time` for default parameters;
- 3D mesh generation returning consistent array sizes.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: FAIL because the tested functions do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- simulation parameters/result dataclasses;
- bridge integration;
- oscillation characteristics;
- full `simulate()` pipeline;
- axisymmetric field/profile extraction;
- 3D mesh generation;
- parameter sweep and summary helpers.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: PASS.

**Step 5: Commit**

Not applicable unless the workspace is later initialized as a git repository.

### Task 3: Build the Streamlit interface

**Files:**
- Create: `droplet_coalescence_3d/web_app.py`

**Step 1: Write the failing test**

Add a smoke test that imports `web_app.py` and calls pure helper functions used to build figures.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: FAIL because helper functions are not implemented.

**Step 3: Write minimal implementation**

Implement:

- simulation tab with parameter controls;
- Plotly 3D surface rendering;
- time-series plots;
- experiment tab with sweep controls, plots, and CSV download;
- model description panel.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: PASS.

**Step 5: Commit**

Not applicable unless the workspace is later initialized as a git repository.

### Task 4: Add headless verification artifacts

**Files:**
- Create: `droplet_coalescence_3d/self_test.py`

**Step 1: Write the failing test**

Add a test that executes the self-test entry point in a temporary directory and expects output artifacts.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: FAIL because the self-test script does not exist or does not generate artifacts.

**Step 3: Write minimal implementation**

Generate:

- one static 3D PNG figure;
- one experiment PNG figure;
- one CSV file with sweep results.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: PASS.

**Step 5: Commit**

Not applicable unless the workspace is later initialized as a git repository.

### Task 5: Verify the end-to-end workflow

**Files:**
- Modify: `droplet_coalescence_3d/README.md`

**Step 1: Run unit and smoke tests**

Run: `python3 -m pytest droplet_coalescence_3d/test_core_3d.py -q`
Expected: PASS.

**Step 2: Run the headless self-test**

Run: `python3 droplet_coalescence_3d/self_test.py`
Expected: output artifacts created without GUI errors.

**Step 3: Verify Streamlit app imports cleanly**

Run: `python3 -c "import sys; sys.path.append('droplet_coalescence_3d'); import web_app"`
Expected: exit code `0`.

**Step 4: Update README with exact launch steps**

Document venv setup, dependency installation, and Streamlit launch command.

**Step 5: Commit**

Not applicable unless the workspace is later initialized as a git repository.
