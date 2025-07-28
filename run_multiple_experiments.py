import subprocess

# Define the experiment runs
start = 0
end = 0
runs = [f"flower=run{i}" for i in range(start, end+1)]

# Run each one sequentially
for run in runs:
    print(f"\n🚀 Starting {run} ...")
    result = subprocess.run(["python3", "run_flower_experiment.py", run])
    if result.returncode != 0:
        print(f"❌ {run} failed with return code {result.returncode}")
    else:
        print(f"✅ {run} completed successfully")

# Run this via command line: Less Overhead and memory usage
#  python3 run_multiple_experiments.py
