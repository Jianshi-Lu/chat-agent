"""验证本地 Qwen3 Embedding 模型的中文政策语义排序。"""

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_PATH = Path(
    "/Users/lujianshi/Desktop/models/embedding/Qwen:Qwen3-Embedding-8B"
)
QUERY = "政府采购绿色建材有哪些政策要求？"
DOCUMENTS = [
    "财政部发布政府采购支持绿色建材促进建筑品质提升政策，明确绿色采购范围和实施要求。",
    "本周末上海天气以多云为主，早晚温差较大，出行请关注天气预报。",
    "交通运输部门发布公路养护项目管理办法，规范项目立项、招标和验收流程。",
]


def main() -> None:
    """加载模型、生成向量并检查相关政策是否排名第一。"""

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"本地向量化模型目录不存在：{MODEL_PATH}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"推理设备：{device}")

    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
        local_files_only=True,
    )
    print(f"向量维度：{model.get_embedding_dimension()}")

    query_embedding = model.encode(
        QUERY,
        prompt_name="query",
        normalize_embeddings=True,
        batch_size=1,
    )
    document_embeddings = model.encode(
        DOCUMENTS,
        normalize_embeddings=True,
        batch_size=1,
    )
    scores = model.similarity(query_embedding, document_embeddings)[0]
    ranking = scores.argsort(descending=True).tolist()

    for rank, index in enumerate(ranking, 1):
        print(f"{rank}. score={scores[index]:.6f} | {DOCUMENTS[index]}")


if __name__ == "__main__":
    main()
