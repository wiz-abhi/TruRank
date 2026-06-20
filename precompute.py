import json
import pickle
import time
from pathlib import Path
from tqdm import tqdm

from src.profile_parser import ProfileParser
from src.embeddings import EmbeddingEngine


def run_precompute(input_path: str, output_path: str, batch_size: int = 512):
    print(f"Starting pre-computation from {input_path}")
    start_time = time.time()

    parser = ProfileParser()
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")

    profiles = []

    print(f"Reading and parsing {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        if str(input_path).endswith(".jsonl"):
            for line in tqdm(f):
                if not line.strip():
                    continue
                try:
                    raw_dict = json.loads(line)
                    profile = parser.parse(raw_dict)
                    profiles.append(profile)
                except Exception as e:
                    print(f"Error parsing line: {e}")
        else:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            for raw_dict in tqdm(data):
                try:
                    profile = parser.parse(raw_dict)
                    profiles.append(profile)
                except Exception as e:
                    print(f"Error parsing item: {e}")

    print(f"Parsed {len(profiles)} profiles in {time.time() - start_time:.2f}s")

    # Pre-calculate texts for embedding
    texts = [p.to_embedding_text() for p in profiles]

    print(f"Generating embeddings for {len(texts)} texts...")
    emb_start = time.time()
    embeddings = engine.model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"Embeddings generated in {time.time() - emb_start:.2f}s")

    # Save cache
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "profiles": profiles,
        "embeddings": embeddings,
        "model": "all-MiniLM-L6-v2",
    }

    print(f"Saving cache to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"Pre-computation finished in {time.time() - start_time:.2f}s total.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="data/raw/candidates.jsonl")
    parser.add_argument("--out", type=str, default="data/processed/candidates_cache.pkl")
    args = parser.parse_args()
    run_precompute(args.candidates, args.out)
