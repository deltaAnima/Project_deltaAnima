from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer
from collections import deque
from pathlib import Path
import numpy as np
import faiss
import datetime
import torch  # to check GPU availablity
import json


"""
    Pydantic class declaration

    Classes: StateTokens, GeneralMem, dialog
"""

"""
    Variables for pydantic
"""
# Max&Min Tokens
MAXTOKEN_LIMIT: int = 100
MINTOKEN_LIMIT: int = 0
# Default state for StateTokens
DEFAULT_STRESS: int = 0
DEFAULT_REWARD: int = 0
# Default state of the GeneralMem
DEFAULT_IMPRESS: int = 40


class StateTokens(BaseModel):
    """
    Container for stress and reward Tokens
    Defines Reminh's stress and dopamine level

    variables:
        stress : int --> Field(0~100)
        reward : int --> Field(0~100)
    """

    stress: int = Field(default=DEFAULT_STRESS, ge=MINTOKEN_LIMIT, le=MAXTOKEN_LIMIT)
    reward: int = Field(default=DEFAULT_REWARD, ge=MINTOKEN_LIMIT, le=MAXTOKEN_LIMIT)


class dialog(BaseModel):
    """
    Container for dialog or monolog of Reminh
    """

    Reminh_status: Optional[str]
    Reminh_text: Optional[str]
    User_name: Optional[str]
    User_text: Optional[str]
    image_mem: Optional[List[str]]


class GeneralMem(BaseModel):
    """
    Container of Memory

    variables:
        id : int            --> id of the memory
        context : dialog    --> dialog or monolog of the device_count

    """

    id: int
    context: dialog
    timestamp: str
    impressiveness: int = Field(
        default=DEFAULT_IMPRESS, ge=MINTOKEN_LIMIT, le=MAXTOKEN_LIMIT
    )
    emotions_analysis: Dict[str, Any]
    state_tokens: Optional[StateTokens] = None


"""
    The class that handles Reminh's memory.

        1. memory queue
            -> contains recent raw memory of converstations
        2. vector db
            -> if memory popped from the memory queue, automatically emebedd and saved.
               it also retrive vector db with faiss to find relative memory with input.
:with
    This is RAG 1.0 but better. (bge-m3's performence is better then last one')

    Name reference from HSR
"""


class Fuli:
    """
    Constructor method of Fuli
    """

    def __init__(
        self, mem_queue_len: int = 10, top_k: int = 5, mem_update_cnt: int = 1
    ):
        # Check GPU availability (no raise, just warn and fallback to CPU)
        self.device = self.__check_gpu_avail()

        # General variable
        self.k = top_k
        self.mem_update_cnt = mem_update_cnt
        self.memory_cnt = 0

        # Class building starts from here
        # 1. Generate memory layer -------------------------------------------------------------------------------
        #   i. Recent memories (deque for queue)
        self.memory_queue = deque(maxlen=mem_queue_len)

        #   ii. Embedding dim and model
        self.embedding_dim: int = 1024  # bge-m3 dense dim
        self.project_root = Path(__file__).resolve().parent
        model_path: str = "/home/user/mnt/nvme_db/models/embed/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedding_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.embedding_model.eval()

        #   iii. FAISS index(cosine similarity with IP)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        #   iv. Stored memories (list of GeneralMem, for retrieval)
        self.memories: List[GeneralMem] = []

        #   v. Load DBs (expanduser for ~ handling)
        self.general_DB_path = Path(
            "/home/user/work/Reminh/Reminh_mem/initial_version/GDB"
        ).resolve()
        self.vector_DB_path = Path(
            "/home/user/work/Reminh/Reminh_mem/initial_version/VDB"
        ).resolve()
        self.general_DB_path.mkdir(parents=True, exist_ok=True)
        self.vector_DB_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.vector_DB_path / "faiss_index.index"
        self.memories_file = self.general_DB_path / "memories.json"

        #   Load existing DB
        self.__load_db()
        # -------------------------------------------------------------------------------------------------------------

        # TODO: ADD new deltaEGO

    def __check_gpu_avail(self) -> str:
        """
        This function checks gpu availablity in the constructor.
        It will act like private method

        Returns
        """
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs available: {gpu_count}")
            for i in range(gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return "cuda:1"  # using 3090 (cuda:0 = 5090, cuda:1 = 3090)
        else:
            print("No GPU detected. Falling back to CPU.")
            return "cpu"

    def __embed_text(self, text: str) -> np.ndarray:
        """Embed text using bge-m3 dense"""
        with torch.no_grad():
            encoded_input = self.tokenizer(
                [text], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.embedding_model(**encoded_input)

            emb = model_output[0][:, 0]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2 Normalization

        return emb.cpu().numpy().astype("float32")

    def add_memory(
        self,
        user_in: str,
        user_name: str,
        AI_response: str,
        AI_status: str,
        deltaEGO_analysis: Dict[str, Any],
        image: List[str] = ["No image"],
    ) -> None:
        """Add new memory: to queue, if popped, to FAISS"""
        # Building memory
        current_memory: dialog = dialog(
            Reminh_status=AI_status,
            Reminh_text=AI_response,
            User_name=user_name,
            User_text=user_in,
            image_mem=image,
        )
        # TODO: ADD EMOTION HERE
        new_mem = GeneralMem(
            id=-1,  # Shows it won't be put in the db yet
            context=current_memory,
            timestamp=datetime.datetime.now().isoformat(),
            impressiveness=50,
            emotions_analysis=deltaEGO_analysis,
            state_tokens=StateTokens(stress=30, reward=70),
        )
        self.memory_queue.append(new_mem)

        # If queue full, pop oldest and add to vector DB
        if self.memory_queue.maxlen is not None and len(self.memory_queue) >= int(
            self.memory_queue.maxlen
        ):
            popped_memory: GeneralMem = self.memory_queue.popleft()

            # Embedding memory
            # It will embed both user and Ai's context
            embed_text = f"User: {popped_memory.context.User_text} AI: {popped_memory.context.Reminh_text}"
            embedding_result = self.__embed_text(embed_text)

            # save embedding into FAISS
            self.index.add(embedding_result)

            popped_memory.id = len(self.memories)
            self.memories.append(popped_memory)
            self.memory_cnt += 1

            # does memory needs to be updated? (in db path as a file of .index and .json)
            if self.memory_cnt >= self.mem_update_cnt:
                self.save_db()
                self.memory_cnt = 0

    def retrieve(self, query: str, k: int | None = None) -> str:
        """
        Retrieve recent memories (from queue) AND top-k relevant long-term memories (from DB).
        Returns: [Recent Memories] + [Retrieved Past Memories]
        """
        recent_memories = list(self.memory_queue)

        long_term_memories = []
        if self.index.ntotal > 0:
            q_emb = self.__embed_text(query)
            distances, indices = self.index.search(q_emb, k=(k or self.k))
            for i, idx in enumerate(indices[0]):
                if idx != -1 and distances[0][i] >= 0.7:
                    long_term_memories.append(self.memories[idx])

        # --- Refomat ---
        memories = []

        # 1. Long Term
        if long_term_memories:
            memories.append("[Reminh's Past Memories]")
            for m in long_term_memories:
                # timestamp(ex: 2026-04-02)
                date = m.timestamp.split("T")[0]
                status_tag = (
                    f"(*{m.context.Reminh_status}*)" if m.context.Reminh_status else ""
                )
                memories.append(
                    f"- ({date}) {m.context.User_name}: {m.context.User_text} | Reminh: ({status_tag}){m.context.Reminh_text}"
                )

        # 2. Recent
        if recent_memories:
            memories.append("\n[Recent Conversation Flow]")
            for m in recent_memories:
                status_tag = (
                    f"(*{m.context.Reminh_status}*)" if m.context.Reminh_status else ""
                )
                memories.append(
                    f"{m.context.User_name}: {m.context.User_text} -> Reminh: {m.context.Reminh_text}"
                )

        return "\n".join(memories)

    def save_db(self):
        """Save FAISS index and memories"""
        faiss.write_index(self.index, str(self.index_file))
        # Pydantic -> dict
        memories_dict = [mem.model_dump() for mem in self.memories]

        with open(self.memories_file, "w", encoding="utf-8") as f:
            json.dump(memories_dict, f, ensure_ascii=False, indent=2)
        print("DB saved.")

    def __load_db(self):
        """Load FAISS index and memories"""
        if self.index_file.exists() and self.memories_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.memories_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # dict -> Pydantic
                self.memories = [GeneralMem.model_validate(d) for d in data]
            print("DB loaded.")
        else:
            print("No existing DB. Starting fresh.")
