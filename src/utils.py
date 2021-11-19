from src.models.embedding import GloveEmbedding


def load_embeddings_from_path(dimension, embed_type, embed_path, trainable = False):
    if embed_type == "glove":
        embed_path = embed_path.format(dims = dimension)
        embedding = GloveEmbedding(400001, dimension, {}, trainable)
        embedding.load_embeddings(embed_path)
        return embedding

    else:
        raise NotImplementedError("Embedding not implemented.")
