from __future__ import annotations

import os

from rq import Connection, Worker

from task_queue.rq_client import get_redis_connection


if __name__ == '__main__':
    conn = get_redis_connection()
    queue_names = os.getenv('RQ_QUEUES', 'default').split(',')
    with Connection(conn):
        worker = Worker(queue_names)
        worker.work(with_scheduler=True)
