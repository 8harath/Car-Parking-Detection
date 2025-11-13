"""
Setup script to create necessary directory structure for the application
"""
import os
import config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create all necessary directories for the application"""
    directories = [
        config.REPORTS_DIR,
        config.DATA_DIR,
        config.MODELS_DIR
    ]

    logger.info("Setting up directory structure...")

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✓ Created/verified directory: {directory}")
        except Exception as e:
            logger.error(f"✗ Error creating directory {directory}: {e}")
            raise

    logger.info("Directory setup complete!")
    print("\n✓ All directories are ready!")
    print(f"  - Reports will be saved to: {config.REPORTS_DIR}")
    print(f"  - Data files will be saved to: {config.DATA_DIR}")
    print(f"  - Models will be stored in: {config.MODELS_DIR}")

if __name__ == "__main__":
    setup_directories()
