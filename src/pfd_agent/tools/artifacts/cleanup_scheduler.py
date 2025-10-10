"""
Scheduled cleanup utility for uploaded files
Can be run manually or as a cron job
"""
import time
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_old_files(directory: Path, max_age_hours: int = 24) -> int:
    """
    Delete files older than max_age_hours from the directory.

    Args:
        directory: Path to the directory to clean
        max_age_hours: Maximum age of files to keep (in hours)

    Returns:
        Number of files deleted
    """
    if not directory.exists():
        logger.info(f"Directory does not exist: {directory}")
        return 0

    now = datetime.now()
    cutoff_time = now - timedelta(hours=max_age_hours)
    deleted_count = 0

    for file_path in directory.glob('*'):
        if file_path.is_file():
            # Get file modification time
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

            if file_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path.name}: {e}")

    # Try to remove directory if empty
    if deleted_count > 0:
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                logger.info(f"Removed empty directory: {directory}")
        except OSError:
            pass  # Directory not empty

    return deleted_count


def cleanup_all_old_files(max_age_hours: int = 24):
    """Clean up old files from uploaded_files directory"""
    base_dir = Path.cwd()
    uploaded_files_dir = base_dir / 'uploaded_files'

    logger.info(f"Starting cleanup (max age: {max_age_hours} hours)")
    deleted = cleanup_old_files(uploaded_files_dir, max_age_hours)
    logger.info(f"Cleanup complete. Deleted {deleted} files.")
    return deleted


def run_periodic_cleanup(interval_hours: int = 24, max_age_hours: int = 24):
    """
    Run cleanup periodically.

    Args:
        interval_hours: How often to run cleanup (in hours)
        max_age_hours: Maximum age of files to keep (in hours)
    """
    logger.info(f"Starting periodic cleanup service")
    logger.info(f"Interval: {interval_hours} hours, Max age: {max_age_hours} hours")

    while True:
        try:
            cleanup_all_old_files(max_age_hours)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

        # Sleep until next cleanup
        sleep_seconds = interval_hours * 3600
        logger.info(f"Next cleanup in {interval_hours} hours")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up old uploaded files")
    parser.add_argument(
        "--max-age",
        type=int,
        default=24,
        help="Maximum age of files to keep (in hours)"
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Run periodic cleanup (default: run once)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Cleanup interval in hours (for periodic mode)"
    )

    args = parser.parse_args()

    if args.periodic:
        run_periodic_cleanup(args.interval, args.max_age)
    else:
        deleted = cleanup_all_old_files(args.max_age)
        print(f"✅ Cleanup complete. Deleted {deleted} files.")
