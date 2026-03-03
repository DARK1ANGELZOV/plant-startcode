from __future__ import annotations

from pathlib import Path

from rq.job import Job

from task_queue.rq_client import get_queue, get_redis_connection
from task_queue.tasks import run_blind_eval_job, run_finetune_job


class JobService:
    def __init__(self, queue_name: str = 'default') -> None:
        self.queue = get_queue(queue_name)
        self.redis = get_redis_connection()
        self.project_root = str(Path(__file__).resolve().parents[1])

    def enqueue_finetune(
        self,
        config_path: str,
        data_yaml: str,
        epochs: int,
        batch: int,
        imgsz: int,
        name: str,
    ) -> str:
        job = self.queue.enqueue(
            run_finetune_job,
            config_path,
            data_yaml,
            epochs,
            batch,
            imgsz,
            name,
            self.project_root,
            job_timeout='24h',
        )
        return job.id

    def enqueue_blind_eval(
        self,
        data_yaml: str,
        split: str,
        max_images: int,
        iou_sla: float,
    ) -> str:
        job = self.queue.enqueue(
            run_blind_eval_job,
            data_yaml,
            split,
            max_images,
            iou_sla,
            self.project_root,
            job_timeout='4h',
        )
        return job.id

    def status(self, job_id: str) -> dict:
        job = Job.fetch(job_id, connection=self.redis)
        error = None
        if job.is_failed:
            error = str(job.exc_info)[-2000:] if job.exc_info else 'Job failed.'
        return {
            'job_id': job.id,
            'status': job.get_status(refresh=True),
            'result': job.result,
            'error': error,
        }
