import hashlib

def hash_snippet(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()

class IncrementalIndexer:
    def __init__(self):
        self.snippet_hashes = {}
        self.index = []

    def add_or_update(self, id, code):
        code_hash = hash_snippet(code)
        if id not in self.snippet_hashes or self.snippet_hashes[id] != code_hash:
            self.snippet_hashes[id] = code_hash
            # Imagine we embed and add to FAISS index here
            self.index.append(id)
            return True
        return False
