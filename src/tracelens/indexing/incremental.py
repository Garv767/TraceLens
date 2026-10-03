import hashlib
import json
import os

def hash_snippet(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()

class IncrementalIndexer:
    def __init__(self, cache_file=".tracelens_cache.json"):
        self.cache_file = cache_file
        self.snippet_hashes = {}
        self.index = []
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.snippet_hashes = json.load(f)

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.snippet_hashes, f)

    def add_or_update(self, id, code):
        code_hash = hash_snippet(code)
        if id not in self.snippet_hashes or self.snippet_hashes[id] != code_hash:
            self.snippet_hashes[id] = code_hash
            self.index.append(id)
            self._save_cache()
            return True
        return False
