# Setup Guide

You said you're new to running Python locally, so this walks through everything from zero. Pick the section for your operating system.

If at any point a command fails, copy the error message — happy to help debug.

---

## Step 1: Install Python

You need Python **3.10 or newer**. Check what you have:

```bash
python3 --version
```

If you see `Python 3.10.x` or higher, skip to Step 2. Otherwise:

### macOS
The easiest path is [Homebrew](https://brew.sh/). Install Homebrew if you don't have it (one command on their homepage), then:

```bash
brew install python@3.12
```

### Windows
Download the installer from [python.org/downloads](https://www.python.org/downloads/). **Important:** during install, tick the box "Add Python to PATH". Then open a fresh PowerShell window.

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

---

## Step 2: Get the project onto your computer

If I've sent you a zip, unzip it somewhere sensible — e.g., `~/projects/portfolio_tool` on Mac/Linux, or `C:\projects\portfolio_tool` on Windows.

Open a terminal **in that folder**:

- **macOS:** Right-click the folder in Finder → "New Terminal at Folder" (you may need to enable this in System Settings → Keyboard → Keyboard Shortcuts → Services).
- **Windows:** In File Explorer, click the address bar, type `cmd`, press Enter.
- **Linux:** Right-click the folder → "Open Terminal Here".

Verify you're in the right place — running `ls` (Mac/Linux) or `dir` (Windows) should show `portfolio/`, `notebooks/`, `requirements.txt`, etc.

---

## Step 3: Create a virtual environment

A virtual environment is a self-contained Python install just for this project. It keeps the project's libraries from conflicting with anything else on your system. Skip this step at your peril.

From inside the project folder:

```bash
python3 -m venv .venv
```

This creates a `.venv/` folder. Activate it:

- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`

Once activated, your prompt will start with `(.venv)`. **You need to re-run this command every time you open a new terminal for this project.**

---

## Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

This will take a couple of minutes. `cvxpy` pulls in some compiled solvers; that's normal.

**NB**: However for most Windows Installs with Python 3.12 or 3.13 this library will fail.

Solution: you need to install: https://aka.ms/vs/17/release/vc_redist.x64.exe
And restart your machine

---

## Step 5: Verify it works

```bash
python tests/test_pipeline.py
```

You should see a long output ending in `ALL TESTS PASSED`. If so, you're done.

---

## Step 6: Open the notebook

```bash
jupyter notebook notebooks/01_walkthrough.ipynb
```

Your browser will open to the notebook. Run cells with **Shift+Enter**. The notebook walks you through fetching real data, building portfolios, and running a backtest.

---

## Common issues

**"yfinance: HTTPError 401"**: Yahoo occasionally throttles. Wait a minute, try again. The cache is your friend — once data is fetched, it sits in `cache/` for 7 days.

**"cvxpy: solver failed"**: usually means the inputs are pathological (a non-PSD covariance, all weights forced to zero, etc.). Check that your covariance matrix has no NaN values and that bucket targets sum to 1.0.

**"ModuleNotFoundError: No module named 'portfolio'"**: you're either not in the project root, or the virtual environment isn't active. Run `which python` (Mac/Linux) or `where python` (Windows) — it should point inside `.venv/`.

**Anything else**: paste the error and ask. I'd rather you ask than fight a confusing error alone.
