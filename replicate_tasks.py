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
            "Transform the person in the uploaded photo into an korean adult version, around 25 years old. "
            "Make the face look slightly more mature while keeping the same identity, gender, and hairstyle. "
            "Add subtle adult facial proportions and natural skin texture without wrinkles. "
            "Render as a realistic, high-quality studio portrait with soft lighting and neutral background."
        )

        self.prompt_young = (
            "Make the person look like a youthful Korean young adult with a fresh, lively appearance. "
            "Keep the same facial identity and gender, but remove wrinkles and signs of aging. "
            "Smooth the skin naturally and brighten the eyes for a healthy, energetic expression. "
            "Give the look and vibe of a college-age person in Korea, with soft clear skin and natural dark hair. "
            "Maintain realistic proportions and a natural facial structure. "
            "Render as a photorealistic high-quality portrait under soft daylight."
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
        self,
        inputs,  # [aged_bytes_or_url, orig_bytes_or_url]
        pose_prompt: str,
        index: int,
        token: str,
        seed: int = 42,
        aspect_ratio: str = "1:1",
        resolution: str = "720p",
    ):
        super().__init__()
        self.inputs = inputs
        self.pose_prompt = pose_prompt
        self.index = index
        self.seed = seed
        self.token = token
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution
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
            refs = self._normalize_image_inputs(self.inputs)

            out = replicate.run(
                "runwayml/gen4-image",
                input={
                    "prompt": self.pose_prompt,
                    "reference_images": refs,  # 최대 3장
                    "reference_tags": ["aged", "orig"],  # refs와 같은 순서
                    "aspect_ratio": self.aspect_ratio,  # 예: "1:1"
                    "resolution": self.resolution,  # 예: "1080p"
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
