from typing import Annotated, TypedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select, delete
from sqlalchemy.exc import NoResultFound

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


class ArxivMapping(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    arxiv_id: str = Field(index=True)


class ArxivEntry(BaseModel):
    arxiv_id: str
    content: str


class ArxivSearchQuery(BaseModel):
    queries: list[str]
    max_results: int


class ArxivSearchResult(BaseModel):
    arxiv_id: str
    score: float


# Database
sqlite_file_name = DATA_DIR / "db.sqlite3"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(
    "./pretrained_models/Alibaba-NLP/gte-large-en-v1.5",
    device=device,
    local_files_only=True,
    trust_remote_code=True,
)
model_dim = model.get_sentence_embedding_dimension()


class EmbeddingStore(TypedDict):
    arxiv: torch.Tensor


embedding_file_name = DATA_DIR / "embeddings.pt"
embedding_store: EmbeddingStore


def load_embeddings():
    global embedding_store
    try:
        embedding_store = torch.load(
            embedding_file_name, map_location=device, weights_only=True
        )
    except FileNotFoundError:
        embedding_store = {}

    if "arxiv" in embedding_store:
        assert embedding_store["arxiv"].dim() == 2
        assert embedding_store["arxiv"].size(1) == model_dim
    else:
        embedding_store["arxiv"] = torch.empty(
            0, model_dim, dtype=torch.float32, device=device
        )


def save_embeddings():
    torch.save(embedding_store, embedding_file_name)


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


# App
app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    load_embeddings()


@app.on_event("shutdown")
def on_shutdown():
    save_embeddings()


@app.post("/arxiv/batch")
def batch_add_arxiv_entries(entries: list[ArxivEntry], session: SessionDep) -> None:
    """
    批量添加或更新 arXiv 论文条目。
    """
    # Find the ids for the entries that already exist
    arxiv_ids = [entry.arxiv_id for entry in entries]
    existing_entries = session.exec(
        select(ArxivMapping).where(ArxivMapping.arxiv_id.in_(arxiv_ids))
    ).all()

    mappings = {entry.arxiv_id: entry for entry in existing_entries}

    # Allowing for the case where the entry already exists
    new_entries = [
        ArxivMapping(arxiv_id=entry.arxiv_id)
        for entry in entries
        if entry.arxiv_id not in mappings
    ]
    session.add_all(new_entries)
    session.commit()

    mappings.update({entry.arxiv_id: entry for entry in new_entries})

    # Encode the content
    contents = [entry.content for entry in entries]
    embeddings = model.encode(contents, convert_to_tensor=True, show_progress_bar=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Update the embeddings
    indices = torch.tensor(
        [mappings[entry.arxiv_id].id for entry in entries], device=device
    )
    embedding_store["arxiv"] = expand_and_update_embeddings(
        embedding_store["arxiv"], embeddings, indices
    )


@app.post("/arxiv/search")
def search_arxiv_entries(
    payload: ArxivSearchQuery, session: SessionDep
) -> list[list[ArxivSearchResult]]:
    """
    搜索 arXiv 论文条目。
    """
    query_embeddings = model.encode(payload.queries, convert_to_tensor=True)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    arxiv_embeddings = embedding_store["arxiv"]
    scores = query_embeddings @ arxiv_embeddings.T

    results: list[list[ArxivSearchResult]] = []

    for score in scores:
        top_scores, top_indices = score.topk(min(payload.max_results, len(score)))
        top_scores = top_scores.tolist()
        top_indices = top_indices.tolist()

        search_results: list[ArxivSearchResult] = []
        for score, index in zip(top_scores, top_indices):
            try:
                entry = session.exec(
                    select(ArxivMapping).where(ArxivMapping.id == index)
                ).one()
            except NoResultFound:
                continue
            search_results.append(
                ArxivSearchResult(arxiv_id=entry.arxiv_id, score=score)
            )

        results.append(search_results)

    return results


@app.post("/arxiv/clear")
def clear_arxiv_entries(session: SessionDep) -> None:
    """
    清空 arXiv 论文条目。
    """
    session.exec(delete(ArxivMapping))
    embedding_store["arxiv"] = torch.empty(
        0, model_dim, dtype=torch.float32, device=device
    )
    save_embeddings()
