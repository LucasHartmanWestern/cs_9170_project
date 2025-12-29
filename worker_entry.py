# worker_entry.py
import json, sys
from worker import worker

def main():
    gpu_id = int(sys.argv[1])
    jobs_file = f"gpu{gpu_id}_jobs.json"
    with open(jobs_file, "r") as f:
        jobs = json.load(f)
    worker(jobs, gpu_id)

if __name__ == "__main__":
    main()
