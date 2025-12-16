# Clone FinCast GitHub page
git clone https://github.com/vincent05r/FinCast-fts

# Rename the folder so that we can access it as a package in our scripts
mv FinCast-fts/ fincast/

# Rename the package from my_project to fincast
sed -i '' "s/name=\"my_project\"/name=\"fincast\"/g" fincast/setup.py

# Make pyprpoject.toml file
cat > fincast/pyproject.toml <<'TOML'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fincast"
version = "0.1.0"
description = "FinCast local package"
readme = "README.md"
requires-python = ">=3.8"
dependencies = []
TOML

uv add --editable fincast/

# Download model weights
mkdir -p weights
wget https://huggingface.co/Vincent05R/FinCast/resolve/main/v1.pth -O weights/v1.pth