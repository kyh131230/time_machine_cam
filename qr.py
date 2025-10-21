import io, html, time, hashlib, requests, qrcode
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
from PyQt5.QtCore import Qt


class QRCODE:
    def __init__(self):
        self.TITLE = "세대 체인지 AI 인생사진관"
        self.session = requests.Session()  # keep-alive로 약간 더 빠르게
        self._cache = {}  # {sha256: (page_url, qr_path, ts)}
        self.HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body>
<a id="dl" href="{file_url}" download="{suggest_name}">download</a>
<script>
  const a=document.getElementById('dl'); a.click();
</script>
</body></html>"""

    # --- QPixmap -> bytes 저장 헬퍼 ---
    @staticmethod
    def _save_qpixmap(pm: QPixmap, fmt: str = "PNG", quality: int = -1) -> bytes:
        qba = QByteArray()
        buf = QBuffer(qba)
        buf.open(QIODevice.WriteOnly)
        pm.save(buf, fmt, quality)  # quality는 JPEG/WebP에서만 의미 있음
        buf.close()
        return bytes(qba)

    @staticmethod
    def _downscale(pm: QPixmap, max_side: int = 1080) -> QPixmap:
        w, h = pm.width(), pm.height()
        if max(w, h) <= max_side:  # 이미 작으면 그대로
            return pm
        if w >= h:
            nw, nh = max_side, int(h * (max_side / w))
        else:
            nh, nw = max_side, int(w * (max_side / h))
        return pm.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def upload_to_0x0st(self, file_bytes: bytes, filename: str) -> str:
        files = {"file": (filename, io.BytesIO(file_bytes))}
        last_err = None
        for attempt in range(3):  # 가벼운 재시도
            try:
                r = self.session.post(
                    "https://0x0.st",
                    files=files,
                    timeout=(5, 40),
                    headers={"User-Agent": "qr-uploader/1.0"},
                )
                r.raise_for_status()
                url = r.text.strip()
                if url.startswith("https://0x0.st/"):
                    return url
                raise RuntimeError(
                    f"0x0.st upload failed: {r.status_code} {r.text[:200]!r}"
                )
            except requests.exceptions.RequestException as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        raise last_err if last_err else RuntimeError("0x0.st upload unknown error")

    def make_qr_png(self, url: str, out_path: str = "qr_download.png") -> str:
        qr = qrcode.QRCode(border=2, box_size=8)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(out_path)
        print(f"📷 QR saved → {out_path}")
        return out_path

    def run(self, pm_or_bytes, mode: str = "fast") -> tuple[str, str]:
        """
        mode: "fast" = 업로드 1회(압축 JPEG/WebP) + 이미지 URL QR
              "html" = 이미지 + HTML 2회 업로드(파일명 제안 필요할 때)
        return: (qr_path, page_url)
        """
        # --- 입력 정규화 ---
        if isinstance(pm_or_bytes, QPixmap):
            pm = pm_or_bytes
        elif isinstance(pm_or_bytes, (bytes, bytearray)):
            # bytes로 들어오면 QPixmap에 로드해 다운스케일/재인코딩 가능하게
            pm = QPixmap()
            pm.loadFromData(bytes(pm_or_bytes))
        else:
            raise TypeError("run() expects QPixmap or bytes")

        # --- 다운스케일 + JPEG로 압축(대폭 빨라짐) ---
        pm_small = self._downscale(pm, 1080)
        jpg_bytes = self._save_qpixmap(pm_small, "JPG", quality=85)

        # --- 캐시 체크(같은 이미지면 재사용) ---
        key = hashlib.sha256(jpg_bytes).hexdigest()
        rec = self._cache.get(key)
        if rec and time.time() - rec[2] < 3600:  # 1시간 캐시
            page_url, qr_path, _ = rec
            return qr_path, page_url

        # --- 업로드 (fast: 1회 / html: 2회) ---
        img_url = self.upload_to_0x0st(jpg_bytes, "lifephoto.jpg")

        if mode == "html":
            try:
                html_bytes = self.HTML_TEMPLATE.format(
                    title=html.escape(self.TITLE),
                    file_url=html.escape(img_url),
                    suggest_name=html.escape("세대_체인지_AI_인생사진관.jpg"),
                ).encode("utf-8")
                page_url = self.upload_to_0x0st(
                    html_bytes, "세대_체인지_AI_인생사진관.html"
                )
            except Exception as e:
                print("⚠️ HTML 업로드 실패, 이미지 URL로 폴백:", e)
                page_url = img_url
        else:
            # ✅ 가장 빠름: 이미지 URL 바로 QR
            page_url = img_url

        qr_path = self.make_qr_png(page_url)
        self._cache[key] = (page_url, qr_path, time.time())
        return qr_path, page_url
