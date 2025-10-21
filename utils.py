#!/usr/bin/env python3
import os
def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/videos", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("model", exist_ok=True)
