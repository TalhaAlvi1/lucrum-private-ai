import subprocess
import os

def run_docker_compose():
    project_root = os.path.dirname(os.path.abspath(__file__))  # One level: script folder
    compose_path = os.path.join(project_root, "docker", "docker-compose.yml")
    subprocess.run(["docker-compose", "-f", compose_path, "up", "-d"], check=True)

if __name__ == "__main__":
    print("🚀 Starting EspoCRM Docker container...")
    run_docker_compose()
    print("✅ EspoCRM is running at: http://localhost:8080")
