import os, base64, replicate, requests
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable


class WorkerSignals(QObject):
    age_done = pyqtSignal(str)
    pose_done = pyqtSignal(int, bytes)
    error = pyqtSignal(str)


class AgeJob(QRunnable):

    def __init__(self, inputs, mode: str, token: str, seed: int = 42):
        super().__init__()
        self.inputs = inputs
        self.mode = mode
        self.seed = seed
        self.token = token
        self.signals = WorkerSignals()

        self.prompt_old = (
            "Transform the person in the uploaded photo into an elderly woman, around 70 years old. "
            "Add deep and defined facial wrinkles, pronounced crow’s feet near the eyes, sagging skin around the neck and cheeks, "
            "and visible fine lines across the forehead and mouth area. "
            "Include silver-white hair with a soft, natural texture, slightly thinner and less voluminous. "
            "Add subtle age spots and slightly duller skin tone, keeping realism. "
            "Maintain the same face identity, facial structure, and expression, but clearly show the effects of aging. "
            "Render as a realistic, high-resolution portrait with soft, natural lighting and neutral background."
            "The transformation should be visibly aged and realistic, not subtle."
        )

        self.prompt_young = (
            "Transform the person in the uploaded photo into a youthful version, around 20 to 25 years old. "
            "Make the skin smooth and radiant, remove wrinkles, and slightly enhance facial symmetry and jawline firmness. "
            "Keep the same hairstyle, pose, lighting, and composition. "
            "Maintain facial identity and expression. "
            "Render as a high-quality, photorealistic studio portrait with soft natural light and clean background."
        )

    def _to_data_uri_from_bytes(self, png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode()
        return "data:image/png;base64," + b64

    def run(self):
        try:
            os.environ["REPLICATE_API_TOKEN"] = self.token
            image_input = self._to_data_uri_from_bytes(self.inputs)
            out = replicate.run(
                "google/nano-banana",
                input={
                    "prompt": (
                        self.prompt_old if self.mode == "future" else self.prompt_young
                    ),
                    "image_input": [image_input],
                    "output_format": "jpg",
                    "seed": self.seed,
                },
            )
            url = (
                out.url
                if hasattr(out, "url")
                else (out[0] if isinstance(out, list) else None)
            )
            if not url:
                raise RuntimeError("Replicate output URL을 얻지 못했습니다.")
            self.signals.age_done.emit(url)  # bytes 전달
        except Exception as e:
            self.signals.error.emit(f"[Age {e}")


class PoseJob(QRunnable):
    """
    나이 변환된 결과 이미지 URL(base_url)을 입력으로 포즈 1개 생성.
    완료 시 pose_done(index, bytes) 방출 (메인 스레드에서 QPixmap 생성).
    """

    def __init__(
        self, inputs, pose_prompt: str, index: int, token: str, seed: int = 42
    ):
        super().__init__()
        self.inputs = inputs
        self.pose_prompt = pose_prompt
        self.index = index
        self.seed = seed  # 포즈별 다른 seed
        self.token = token
        self.signals = WorkerSignals()

    def _to_data_uri_from_bytes(self, png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode()
        return "data:image/png;base64," + b64

    def _normalize_image_inputs(self, inputs):
        norm = []
        for item in inputs:
            if isinstance(item, (bytes, bytearray)):
                norm.append(self._to_data_uri_from_bytes(bytes(item)))

            elif isinstance(item, str) and item.startswith(
                ("http://", "https://", "data:image/")
            ):
                norm.append(item)
            else:
                raise ValueError("지원하지 않는 타입")

        return norm

    def run(self):
        try:
            os.environ["REPLICATE_API_TOKEN"] = self.token
            image_input = self._normalize_image_inputs(self.inputs)

            out = replicate.run(
                "flux-kontext-apps/multi-image-kontext-max",
                input={
                    "prompt": self.pose_prompt,
                    "aspect_ratio": "1:1",
                    "input_image_1": image_input[0],
                    "input_image_2": image_input[1],
                    "output_format": "png",
                    "seed": self.seed,
                },
            )
            url = (
                out.url
                if hasattr(out, "url")
                else (out[0] if isinstance(out, list) else None)
            )
            if not url:
                raise RuntimeError("포즈 결과 URL을 얻지 못했습니다.")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            self.signals.pose_done.emit(self.index, resp.content)  # bytes 전달
        except Exception as e:
            self.signals.error.emit(f"[pose {self.index}] {e}")
