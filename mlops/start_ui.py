import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start MLflow UI Server for HybridGAN-BWE")
    parser.add_argument("--port", type=int, default=5000, help="Port to run MLflow UI server on (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for MLflow UI server")
    parser.add_argument("--backend_store_uri", type=str, default="mlruns", help="Directory where MLflow runs are stored")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", args.backend_store_uri,
        "--host", args.host,
        "--port", str(args.port)
    ]

    print(f"[MLOps] Starting MLflow UI at http://{args.host}:{args.port} (backend store: '{args.backend_store_uri}')...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[MLOps] Stopped MLflow UI server.")

if __name__ == "__main__":
    main()
