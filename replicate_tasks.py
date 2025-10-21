import os, base64, replicate, requests
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable
from PyQt5.QtCore import Qt, QByteArray, QBuffer, QIODevice
from PyQt5.QtGui import QImage
import time


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
            "Transform the person in the uploaded photo into an korean adult version, around 25-30 years old Korean college freshman."
            "Make the face look slightly more mature while keeping the same identity, gender, and hairstyle. "
            "Add subtle adult facial proportions and natural skin texture without wrinkles. "
            "Render as a realistic, high-quality studio portrait with soft lighting and neutral background."
        )

        self.prompt_young = (
            "Transform the person into around 20 years old Korean college freshman. "
            "Preserve the same facial identity and gender, but make the face look youthful and full of energy. "
            "Smooth the skin naturally, remove wrinkles, and brighten the eyes for a lively and fresh appearance. "
            "Keep a natural dark hair color with soft, healthy shine. "
            "Give the overall feeling of a bright, friendly, first-year university student in Korea. "
            "Render as a photorealistic, high-quality portrait under gentle daylight, with natural proportions and realistic facial texture."
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

    def _shrink_image_bytes(self, png_bytes: bytes, max_side=1024, quality=85) -> bytes:
        img = QImage.fromData(png_bytes)
        if img.isNull():
            return png_bytes
        w, h = img.width(), img.height()
        if max(w, h) > max_side:
            if w >= h:
                nw, nh = max_side, int(h * (max_side / w))
            else:
                nh, nw = max_side, int(w * (max_side / h))
            img = img.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        img.save(buf, "JPG", quality)  # JPEG로 전송량 축소
        buf.close()
        return bytes(ba)

    def _to_data_uri_from_bytes(self, b: bytes) -> str:
        b_small = self._shrink_image_bytes(b, max_side=1024, quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(b_small).decode()

    def _replicate_run_with_retry(self, payload, tries=3):
        last = None
        for i in range(tries):
            try:
                return replicate.run("runwayml/gen4-image", input=payload)
            except Exception as e:
                if "timed out" in str(e).lower() and i < tries - 1:
                    time.sleep(0.8 * (i + 1))
                    continue
                last = e
                break
        raise last

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

            out = self._replicate_run_with_retry(
                {
                    "prompt": self.pose_prompt,
                    "reference_images": refs,  # 최대 3장
                    "reference_tags": ["personA", "personB"],  # refs와 같은 순서
                    "aspect_ratio": self.aspect_ratio,  # 예: "1:1"
                    "resolution": self.resolution,  # 예: "1080p"
                    "seed": self.seed,
                }
            )
            url = (
                out.url
                if hasattr(out, "url")
                else (out[0] if isinstance(out, list) else None)
            )
            if not url:
                raise RuntimeError("포즈 결과 URL을 얻지 못했습니다.")
            resp = requests.get(url, timeout=(5, 120))
            resp.raise_for_status()
            self.signals.pose_done.emit(self.index, resp.content)  # bytes 전달
        except Exception as e:
            self.signals.error.emit(f"[pose {self.index}] {e}")
