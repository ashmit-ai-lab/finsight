"""Visual search: embed screenshots of each page with a ColPali style model, so tables and charts
are searched as they look, with no text extraction at all."""
import config


def _model_classes(name: str):
    from colpali_engine import models as M
    n = name.lower()
    if "colsmol" in n or "idefics" in n:
        return M.ColIdefics3, M.ColIdefics3Processor
    if "colqwen2.5" in n or "colqwen2_5" in n:
        return M.ColQwen2_5, M.ColQwen2_5_Processor
    if "colqwen2" in n:
        return M.ColQwen2, M.ColQwen2Processor
    return M.ColPali, M.ColPaliProcessor


class VisualRetriever:
    def __init__(self, name: str = config.VISUAL_MODEL, device: str = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cls, proc_cls = _model_classes(name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = model_cls.from_pretrained(name, torch_dtype=dtype, device_map=self.device).eval()
        self.processor = proc_cls.from_pretrained(name)

    def _encode(self, batch) -> list:
        batch = batch.to(self.model.device)
        with self.torch.no_grad():
            emb = self.model(**batch)
        mask = batch["attention_mask"].bool()
        return [e[m].to("cpu", self.torch.float16) for e, m in zip(emb, mask)]  # drop padding tokens

    def embed_pages(self, images: list, batch_size: int = 4) -> list:
        out = []
        for i in range(0, len(images), batch_size):
            out += self._encode(self.processor.process_images(images[i:i + batch_size]))
            print(f"  pages {min(i + batch_size, len(images))}/{len(images)}", end="\r")
        print()
        return out

    def search(self, query: str, page_embs: list, k: int = 20) -> list:
        q = self._encode(self.processor.process_queries([query]))
        scores = self.processor.score_multi_vector([x.float() for x in q], [p.float() for p in page_embs])[0]
        top = self.torch.argsort(scores, descending=True)[:k].tolist()
        return [(i + 1, float(scores[i])) for i in top]  # (1 based page, score)


def index_path(ticker: str):
    d = config.INDEX / "visual"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker}.pt"


def build(ticker: str, retriever: VisualRetriever) -> None:
    import torch
    from .render import page_count, page_image
    images = [page_image(ticker, p) for p in range(1, page_count(ticker) + 1)]
    torch.save(retriever.embed_pages(images), index_path(ticker))


def load(ticker: str) -> list:
    import torch
    return torch.load(index_path(ticker))


def search_as_hits(query: str, ticker: str, retriever, page_embs: list, page_texts: list, k: int = 20) -> list:
    """Same hit format as text search; the text is the page's own text, used only for answering."""
    return [{"ticker": ticker, "page": p, "kind": "page", "score": s, "text": page_texts[p - 1]}
            for p, s in retriever.search(query, page_embs, k)]
