import os
from pathlib import Path
from path_settings import FLOWER_BUILD_DIR, FLOWER_BUILD_FAB_PATH

def build_flower():
    """Builds the Flower App into a .fab file on /work."""
    Path(FLOWER_BUILD_DIR).mkdir(parents=True, exist_ok=True)

    if Path(FLOWER_BUILD_FAB_PATH).exists():
        print(f"🧹 Removing old build: {FLOWER_BUILD_FAB_PATH}")
        Path(FLOWER_BUILD_FAB_PATH).unlink()

    print("🚧 Building Flower App Bundle...")
    # Build the Flower app using the flwr CLI
    # move .fab to remote work directory (build always builds in current dir)
    os.system(f"flwr build --app . && mv ./*.fab {FLOWER_BUILD_FAB_PATH}")
    print(f"✅ Flower App built at {FLOWER_BUILD_FAB_PATH}")


if __name__ == "__main__":
    build_flower()