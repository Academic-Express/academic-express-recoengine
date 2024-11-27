import torch

from .common import OriginKey


class EmbeddingStore:
    def __init__(
        self, file_name: str, model_dim: int, device: torch.device | None = None
    ):
        """
        Initialize the embedding store.

        Args:
            file_name (str): The file name to store the embeddings.
            model_dim (int): The dimension of the model.
            device (torch.device, optional): The device to store the embeddings. Defaults to None.
        """
        self.file_name = file_name
        self.model_dim = model_dim
        self.device = device
        self.store: dict[str, torch.Tensor] = {}

    def load(self):
        try:
            self.store = torch.load(
                self.file_name, map_location=self.device, weights_only=True
            )
        except FileNotFoundError:
            self.store = {}

        for key in OriginKey:
            if key in self.store:
                assert self.store[key].dim() == 2
                assert self.store[key].size(1) == self.model_dim
            else:
                self.store[key] = torch.empty(
                    0, self.model_dim, dtype=torch.float32, device=self.device
                )

    def save(self):
        torch.save(self.store, self.file_name)

    def __getitem__(self, key: OriginKey) -> torch.Tensor:
        return self.store[key.value]

    def __setitem__(self, key: OriginKey, value: torch.Tensor):
        self.store[key.value] = value
