import os
import sys

# Add project root to python path to allow direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingest import run_ingestion_pipeline

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    print(f"Starting ingestion process for files in: {data_dir}")
    
    # Create the directory if it doesn't exist to prevent errors
    os.makedirs(data_dir, exist_ok=True)
    
    run_ingestion_pipeline(data_dir)
    
if __name__ == "__main__":
    main()
