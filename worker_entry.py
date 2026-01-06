# worker_entry.py
import sys
from worker import worker

def main():
    gpu_id = int(sys.argv[1])
    jobs_file = f"gpu{gpu_id}_jobs.json"
    worker(jobs_file, gpu_id)

if __name__ == "__main__":
    main()
