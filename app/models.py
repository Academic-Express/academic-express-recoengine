from sqlmodel import Field, SQLModel

from .common import OriginKey


class BaseMapping(SQLModel):
    index: int | None = Field(default=None, primary_key=True)
    entry_id: str = Field(index=True)


class ArxivMapping(BaseMapping, table=True):
    pass


class GithubMapping(BaseMapping, table=True):
    pass


MAPPING_CLASSES: dict[OriginKey, type[BaseMapping]] = {
    OriginKey.arxiv: ArxivMapping,
    OriginKey.github: GithubMapping,
}
