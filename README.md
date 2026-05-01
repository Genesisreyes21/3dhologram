# 3D Particle Hologram

This is a webcam-to-particle prototype. The browser captures a face frame, sends it to a FastAPI backend, the backend runs the released DECA face reconstruction model, and the frontend renders the returned face point cloud as a glowing hologram.

The app also includes a procedural demo point cloud so the frontend can be tested before DECA is installed.

## Project Structure

- `frontend/` contains the browser interface, webcam flow, ml5 FaceMesh helper data, p5.js particle rendering, and visual controls.
- `backend/` contains the FastAPI server, image upload handling, DECA adapter, mesh-to-point-cloud export utilities, requirements, and local startup script.

## What Is Not Included

This repository does not include:

- the official DECA repository
- the DECA model checkpoint, including `deca_model.tar`
- local `.env` files
- Python virtual environments
- generated captures, outputs, meshes, or cache files

Each user must download DECA and its released checkpoint separately according to the DECA project instructions.

## Requirements

- Python 3.10 or newer
- A separate local clone of the official DECA repository
- The released DECA checkpoint downloaded through DECA's setup script
- A browser with webcam access

## Backend Setup

Create and activate a virtual environment:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Clone DECA outside this repository and install its setup requirements:

```bash
git clone https://github.com/yfeng95/DECA.git
cd DECA
pip install -r requirements.txt
bash fetch_data.sh
```

Create your private environment file:

```bash
cd /path/to/this/repo/backend
cp .env.example .env
```

Edit `backend/.env` with your own local paths:

```env
DECA_ROOT=/path/to/local/DECA
DECA_MODEL_PATH=/path/to/local/DECA/data/deca_model.tar
DECA_DEVICE=auto
DECA_RASTERIZER=standard
DECA_FACE_DETECTOR=fan
```

Do not commit `backend/.env`. It is ignored by Git.

## Run Locally

From the backend folder:

```bash
cd backend
source .venv/bin/activate
set -a
source .env
set +a
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

You can also run:

```bash
cd backend
./start.sh
```

## Environment Variables

- `DECA_ROOT`: local path to your separate DECA repository clone.
- `DECA_MODEL_PATH`: local path to the released `deca_model.tar` checkpoint.
- `DECA_DEVICE`: `auto`, `cpu`, or another PyTorch device string.
- `DECA_RASTERIZER`: DECA rasterizer setting, usually `standard`.
- `DECA_FACE_DETECTOR`: detector setting passed through the backend status/config.

## Publishing Safety

Before committing, confirm that `git status` does not show:

- `.env` or `backend/.env`
- `.venv/`, `venv/`, or other virtual environments
- `__pycache__/`
- `.DS_Store`
- DECA model files, checkpoints, weights, meshes, or generated outputs
- personal absolute computer paths
- API keys, tokens, passwords, or credentials
