from typing import Annotated
import warnings

import torch
import torch.nn.functional as F
from fastapi import Body, Depends, FastAPI, HTTPException, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import NoResultFound
from sqlmodel import delete, select

from .common import DATA_DIR, OriginKey
from .database import SessionDep, create_db_and_tables
from .embeddings import EmbeddingStore
from .models import MAPPING_CLASSES


class Entry(BaseModel):
    entry_id: str
    content: str


class SearchQuery(BaseModel):
    queries: list[str]
    max_results: int


class SearchResult(BaseModel):
    entry_id: str
    score: float


### Authorization

security = HTTPBearer()

try:
    with open(DATA_DIR / "access_token", "r") as f:
        access_token = f.read().strip()
except FileNotFoundError:
    warnings.warn(
        "No access token found. Using the default access token.",
        UserWarning,
    )
    access_token = "feed-engine-token"


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != access_token:
        raise HTTPException(status_code=403, detail="Invalid access token.")


### Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(
    "./pretrained_models/Alibaba-NLP/gte-large-en-v1.5",
    device=device,
    local_files_only=True,
    trust_remote_code=True,
)
model_dim = model.get_sentence_embedding_dimension()


### Embeddings

embedding_file_name = DATA_DIR / "embeddings.pt"
embedding_store = EmbeddingStore(embedding_file_name, model_dim, device=device)


def expand_and_update_embeddings(
    target: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    ratio: float = 1.5,
) -> torch.Tensor:
    """
    Expand the target tensor and update the values at the specified indices.

    Args:
        target (torch.Tensor): The target tensor to update.
        values (torch.Tensor): The values to update.
        indices (torch.Tensor): The indices to update.
        ratio (float): The ratio by which to expand the target tensor.
    """
    min_length = indices.max().item() + 1

    # Expand the embeddings tensor if necessary
    if target.size(0) < min_length:
        new_length = max(min_length, int(target.size(0) * ratio))
        new_target = torch.zeros(
            new_length, target.size(1), dtype=target.dtype, device=target.device
        )
        new_target[: target.size(0)] = target
        target = new_target

    # Update the embeddings
    target[indices] = values
    return target


### App

app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    embedding_store.load()


@app.on_event("shutdown")
def on_shutdown():
    embedding_store.save()


@app.post("/{origin}/batch", dependencies=[Depends(verify_token)])
def batch_add_entries(
    origin: Annotated[OriginKey, Path(title="条目来源")],
    entries: Annotated[list[Entry], Body(title="条目列表")],
    session: SessionDep,
) -> None:
    """
    批量添加或更新条目。
    """
    mapping_cls = MAPPING_CLASSES[origin]

    # Find the ids for the entries that already exist
    entry_ids = [entry.entry_id for entry in entries]
    existing_entries = session.exec(
        select(mapping_cls).where(mapping_cls.entry_id.in_(entry_ids))
    ).all()

    mappings = {entry.entry_id: entry for entry in existing_entries}

    # Allowing for the case where the entry already exists
    new_entries = [
        mapping_cls(entry_id=entry.entry_id)
        for entry in entries
        if entry.entry_id not in mappings
    ]
    session.add_all(new_entries)
    session.commit()

    mappings.update({entry.entry_id: entry for entry in new_entries})

    # Encode the content
    contents = [entry.content for entry in entries]
    embeddings = model.encode(contents, convert_to_tensor=True, show_progress_bar=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Update the embeddings
    indices = torch.tensor(
        [mappings[entry.entry_id].index for entry in entries], device=device
    )
    embedding_store[origin] = expand_and_update_embeddings(
        embedding_store[origin], embeddings, indices
    )


@app.post("/{origin}/search", dependencies=[Depends(verify_token)])
def search_entries(
    origin: Annotated[OriginKey, Path(title="条目来源")],
    payload: Annotated[SearchQuery, Body(title="搜索请求")],
    session: SessionDep,
) -> list[list[SearchResult]]:
    """
    搜索条目。
    """
    mapping_cls = MAPPING_CLASSES[origin]

    query_embeddings = model.encode(payload.queries, convert_to_tensor=True)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    entry_embeddings = embedding_store[origin]
    scores = query_embeddings @ entry_embeddings.T

    results: list[list[SearchResult]] = []

    for score in scores:
        top_scores, top_indices = score.topk(min(payload.max_results, len(score)))
        top_scores = top_scores.tolist()
        top_indices = top_indices.tolist()

        search_results: list[SearchResult] = []
        for score, index in zip(top_scores, top_indices):
            try:
                entry = session.exec(
                    select(mapping_cls).where(mapping_cls.index == index)
                ).one()
            except NoResultFound:
                continue
            search_results.append(SearchResult(entry_id=entry.entry_id, score=score))

        results.append(search_results)

    return results


@app.post("/{origin}/clear", dependencies=[Depends(verify_token)])
def clear_entries(
    origin: Annotated[OriginKey, Path(title="条目来源")], session: SessionDep
) -> None:
    """
    清空条目。
    """
    mapping_cls = MAPPING_CLASSES[origin]

    session.exec(delete(mapping_cls))
    embedding_store[origin] = torch.empty(
        0, model_dim, dtype=torch.float32, device=device
    )
    embedding_store.save()


@app.post("/reload", dependencies=[Depends(verify_token)])
def reload_embeddings() -> None:
    """
    重新加载嵌入。
    """
    embedding_store.load()


@app.post("/save", dependencies=[Depends(verify_token)])
def save_embeddings() -> None:
    """
    保存嵌入。
    """
    embedding_store.save()
