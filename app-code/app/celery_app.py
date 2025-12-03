"""Celery configuration for background tasks."""

from celery import Celery
from app.config import settings

# Create Celery instance
celery_app = Celery(
    "pedestrian_counting",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=['app.tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    'analyze-hourly-los': {
        'task': 'app.tasks.analyze_hourly_los',
        'schedule': 3600.0,  # Every hour
    },
    'cleanup-old-data': {
        'task': 'app.tasks.cleanup_old_data',
        'schedule': 86400.0,  # Daily
    },
    'generate-daily-report': {
        'task': 'app.tasks.generate_daily_report',
        'schedule': 86400.0,  # Daily at midnight
    },
}

if __name__ == '__main__':
    celery_app.start()