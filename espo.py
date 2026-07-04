import subprocess
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
compose_path = os.path.join(project_root, "docker", "docker-compose.yml")

def run_docker_compose():
    subprocess.run(["docker-compose", "-f", compose_path, "up", "-d"], check=True)

def stop_docker_compose():
    subprocess.run(["docker-compose", "-f", compose_path, "down"], check=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        print("🛑 Stopping EspoCRM Docker container...")
        stop_docker_compose()
        print("✅ EspoCRM Docker container stopped.")
    else:
        print("🚀 Starting EspoCRM Docker container...")
        run_docker_compose()
        print("✅ EspoCRM is running at: http://localhost:8080")