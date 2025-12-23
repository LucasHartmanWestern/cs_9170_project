import json, sys
from worker import worker

gpu_id = int(sys.argv[1])
jobs_file = f"gpu{gpu_id}_jobs.json"

with open(jobs_file, "r") as f:
    jobs = json.load(f)

worker(jobs, gpu_id)
