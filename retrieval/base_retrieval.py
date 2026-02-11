
class BaseRetrieval():
    def __init__(self, config):
        self.config = config
    
    def prepare(self, dataset):
        """Prepare the retrieval system with a dataset."""
        pass
    
    def find_top_k(self, query):
        """Find top-k results for the given query."""
        raise NotImplementedError("Subclasses must implement find_top_k()")