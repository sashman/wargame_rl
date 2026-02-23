from pydantic import BaseModel


class TransformerConfig(BaseModel):
    n_layers: int = 4  # number of layers in the transformer
    n_heads: int = 4  # number of attention heads
    embedding_size: int = 128  # size of the embedding vector
    dropout: float = 0.0  # dropout rate -> 0.0 means no dropout
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal: bool = False  # We don't want causal attention for the WarTransformer
    block_size: int = 256  # Maximum sequence length for the transformer
