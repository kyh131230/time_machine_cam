import sys, os, glob, cv2
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtCore import (
    Qt,
    QTimer,
    QObject,
    pyqtSignal,
    QRunnable,
    QThreadPool,
    QRect,
    QMarginsF,
)
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QFontDatabase, QCursor
from setting import FileController
from replicate_tasks import AgeJob, PoseJob
import numpy as np
import json
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtCore import QSizeF, QSize
from qr import QRCODE
from PyQt5.QtCore import QFile, QTextStream


def resource_path(rel_path: str) -> str:
    """
    개발환경과 PyInstaller(onefile/onedir) 실행환경 모두에서
    동일하게 사용할 수 있는 안전한 절대경로 생성기.
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)


def cv2_to_qpixmap(bgr):
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class FrameEditorDialog(QtWidgets.QDialog):
    def __init__(self, base_pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame 영역 조정기")
        self.setModal(True)
        self.base_pixmap = base_pixmap
        self.orig_w, self.orig_h = base_pixmap.width(), base_pixmap.height()

        # ---- 미리보기 크기 결정 (화면의 70% 안쪽, 가로 최대 720px 권장) ----
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        max_w = min(720, int(scr.width() * 0.7))
        max_h = int(scr.height() * 0.8)
        self.scale = min(max_w / self.orig_w, max_h / self.orig_h)
        if self.scale <= 0:
            self.scale = 1.0

        self.view_w = int(self.orig_w * self.scale)
        self.view_h = int(self.orig_h * self.scale)

        # 라벨(미리보기)
        self.label = QtWidgets.QLabel()
        self.label.setFixedSize(self.view_w, self.view_h)
        self.label.setAlignment(Qt.AlignCenter)

        # 최초 미리보기 그림
        self.view_pixmap = self.base_pixmap.scaled(
            self.view_w, self.view_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(self.view_pixmap)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        # 상태
        self.rects = []  # 원본 좌표계(QRect)
        self.master_rect = None
        self.start_pos = None  # 원본 좌표계의 시작점(QPoint)
        self.drag_pos = None  # 원본 좌표계의 현재점(QPoint)

        # 이벤트 바인딩
        self.label.mousePressEvent = self._on_mouse_press
        self.label.mouseMoveEvent = self._on_mouse_move
        self.label.mouseReleaseEvent = self._on_mouse_release

        QtWidgets.QToolTip.showText(
            self.mapToGlobal(self.rect().center()),
            "드래그해서 위/아래 박스 2개를 그리세요.",
            self,
        )

        # 다이얼로그 사이즈
        self.resize(self.view_w + 24, self.view_h + 24)

    # ---- 좌표 변환 헬퍼 (뷰 → 원본) ----
    def _to_orig_pt(self, p: QtCore.QPoint) -> QtCore.QPoint:
        x = int(round(p.x() / self.scale))
        y = int(round(p.y() / self.scale))
        x = max(0, min(self.orig_w - 1, x))
        y = max(0, min(self.orig_h - 1, y))
        return QtCore.QPoint(x, y)

    # ---- 사각형 스케일 (원본 → 뷰) ----
    def _to_view_rect(self, r: QtCore.QRect) -> QtCore.QRect:
        return QtCore.QRect(
            int(round(r.x() * self.scale)),
            int(round(r.y() * self.scale)),
            int(round(r.width() * self.scale)),
            int(round(r.height() * self.scale)),
        )

    def _on_mouse_press(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        if len(self.rects) >= 2:
            self._emit_norm_and_close()
            return
        self.start_pos = self._to_orig_pt(ev.pos())
        self.drag_pos = self.start_pos

    def _on_mouse_move(self, ev):
        if self.start_pos is None:
            return
        self.drag_pos = self._to_orig_pt(ev.pos())
        # 미리보기 갱신 (뷰 좌표로 그림)
        preview = self.view_pixmap.copy()
        p = QPainter(preview)
        p.setPen(Qt.red)
        cur = self._current_rect()
        if cur:
            p.drawRect(self._to_view_rect(cur))
        for rr in self.rects:
            p.drawRect(self._to_view_rect(rr))
        p.end()
        self.label.setPixmap(preview)

    def _on_mouse_release(self, ev):
        if self.start_pos is None:
            return
        self.drag_pos = self._to_orig_pt(ev.pos())
        r = self._current_rect()
        self.start_pos = None
        self.drag_pos = None
        if not r or r.width() <= 0 or r.height() <= 0:
            self.label.setPixmap(self.view_pixmap)
            return

        # 첫 박스 확정
        if not self.rects:
            self.master_rect = r
            self.rects.append(r)
        else:
            # 두 번째 박스는 첫 박스의 x/width/height를 강제 고정
            r.setX(self.master_rect.x())
            r.setWidth(self.master_rect.width())
            r.setHeight(self.master_rect.height())
            self.rects.append(r)
            # 위→아래 정렬 후 종료
            self.rects.sort(key=lambda rr: rr.y())
            self._emit_norm_and_close()
            return

        # 프리뷰 갱신
        preview = self.view_pixmap.copy()
        p = QPainter(preview)
        p.setPen(Qt.red)
        for rr in self.rects:
            p.drawRect(self._to_view_rect(rr))
        p.end()
        self.label.setPixmap(preview)

    def _current_rect(self):
        if self.start_pos is None or self.drag_pos is None:
            return None
        r = QtCore.QRect(self.start_pos, self.drag_pos).normalized()
        # 두 번째 박스는 마스터 기준 정렬(가로/높이 동일)
        if self.master_rect:
            r.setX(self.master_rect.x())
            r.setWidth(self.master_rect.width())
            r.setHeight(self.master_rect.height())
        return r

    def _emit_norm_and_close(self):
        W, H = self.orig_w, self.orig_h
        norms = []
        for rr in self.rects[:2]:
            nx = rr.x() / W
            ny = rr.y() / H
            nw = rr.width() / W
            nh = rr.height() / H
            norms.append((round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)))
        self.norms = norms
        QtWidgets.QApplication.clipboard().setText(str(norms))
        print("✅ frame_boxes_norm:", norms)
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self._frame_boxes_path = os.path.join(
            os.path.dirname(__file__), "frame_boxes.json"
        )

        self.qr = QRCODE()

        self.ai_running = False
        self.poses_left = 0

        self.stacked = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked)

        # ui/*.ui 를 알파벳 순서로 자동 로드 (first.ui, second.ui, ...)
        self.pages = []
        self.captured_png_bytes = None

        self.CANVAS_W = 1181  # px  (100 mm @ 300 DPI)
        self.CANVAS_H = 1748  # px  (148 mm @ 300 DPI)

        self.frame_template_paths = [
            resource_path("img/frame_1.png"),
            resource_path("img/frame_2.png"),
        ]
        self.frame_templates = []
        for p in self.frame_template_paths:
            pm = QPixmap(p)
            if pm.isNull():
                pm = QPixmap(self.CANVAS_W, self.CANVAS_H)
                pm.fill(Qt.black)
            # 템플릿을 정확히 캔버스 크기로 맞춤 (왜곡 방지하려면 Expanding 후 center-crop로 바꿔도 됨)
            pm = pm.scaled(
                self.CANVAS_W,
                self.CANVAS_H,
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
            self.frame_templates.append(pm)

        for ui_path in sorted(glob.glob(resource_path("ui/*.ui"))):
            w = uic.loadUi(ui_path)
            self.stacked.addWidget(w)
            self.pages.append(w)

        if not self.pages:
            QtWidgets.QMessageBox.critical(
                self, "오류", "ui 폴더에 .ui 파일이 없습니다."
            )
            sys.exit(1)

        # 버튼 시그널 연결 (각 페이지에 btnNext/btnBack이 있을 때만 연결)
        for idx, page in enumerate(self.pages):
            btn_next = getattr(page, "btn_next", None)
            btn_back = getattr(page, "btn_back", None)

            if btn_next:
                btn_next.clicked.connect(lambda _, i=idx: self.goto_page(i + 1))
            if btn_back:
                btn_back.clicked.connect(lambda _, i=idx: self.goto_page(i - 1))

        self.frame_boxes_norm = [
            # frame_1: 위/아래
            [(0.077, 0.113, 0.85, 0.425), (0.07, 0.548, 0.86, 0.428)],
            # frame_2
            [(0.077, 0.113, 0.85, 0.425), (0.07, 0.548, 0.86, 0.428)],
        ]

        self.final_composed_pixmap = QPixmap()
        self.qrcode_pixmap = None

        self._load_frame_boxes()

        self.goto_page(0)  # 첫 화면
        self._write_mode_buttons()

        self._setup_capture_page()
        self._setup_pick2_page()
        self._setup_frame_page()
        self._setup_print_page()

        if self.btn_next_on_capture:
            self.btn_next_on_capture.clicked.connect(self._start_ai_pipeline)

        if self.pick2_next_btn:
            self.pick2_next_btn.clicked.connect(
                lambda: (
                    self.goto_page(self.frame_page_index),
                    QTimer.singleShot(
                        0, lambda: self._choose_frame(self.selected_frame_index)
                    ),
                )
            )

        self.pool = QThreadPool().globalInstance()

        self.camera_port = (
            FileController().load_json().get("CAMERA_PORT", "")
        )  ## 카메라 포트 json 추가

        self.replicate_token = (
            FileController().load_json().get("REPLICATE_API_TOKEN", "")
        )
        if self.replicate_token:
            os.environ["REPLICATE_API_TOKEN"] = self.replicate_token

        POSE_PROMPTS = [
            "@personA and @personB stand side by side, both smiling and giving a thumbs-up with one hand. Keep @personA and @personB identical to their references (no merging or replacement). Shoulder-to-shoulder, clear front view, 1:1 framing, natural light.",
            "@personA holds smartphone above head level with the right hand for a selfie, while @personB stands close beside making a V sign with one hand. Both look toward the smartphone. Shoulder-to-shoulder, 1:1 framing. not face and hand distortion",
            "@personA and @personB each use one hand to form a heart shape together. Shoulder-to-shoulder, 1:1 framing.",
        ]

        self.pose_prompts = POSE_PROMPTS

        self._load_stylesheet()

    def _load_stylesheet(self):
        # stylesheet.qss 파일 로드
        self.setStyleSheet("")
        qss_file_path = resource_path("style/stylesheet.qss")
        qss_file = QFile(qss_file_path)
        qss_file.open(QFile.ReadOnly | QFile.Text)
        qss_stream = QTextStream(qss_file)
        qss_stream_all = qss_stream.readAll()
        qss_file.close()

        font_path = resource_path("style/font/Maplestory Light.ttf")  # 폰트 파일 경로
        font_name = self._load_external_font(font_path)

        if font_name:
            # QSS에서 폰트 이름 대체
            qss_stream_all = qss_stream_all.replace("Maple Story", font_name)

            # 기존 스타일 초기화 후 새 스타일 적용
            self.setStyleSheet("")
            self.setStyleSheet(qss_stream_all)

            # 프로그램적으로 폰트 설정 (필요시)
            font = QFont(font_name)
            font.setPointSize(16)
            self.setFont(font)
        else:
            self.setStyleSheet(qss_stream_all)

        ## Cursor 기능 임시 비할성화 ##
        # cursor = QPixmap(resource_path("style/cursor/nyj.png"))
        # orig_w, orig_h = cursor.width(), cursor.height()
        # orig_hot_x, orig_hot_y = 357, 524

        # scaled_w, scaled_h = 64, 64
        # cursor_pixmap = cursor.scaled(
        #     scaled_w,
        #     scaled_h,
        #     Qt.KeepAspectRatio,
        #     Qt.SmoothTransformation,
        # )

        # hot_x = int(orig_hot_x * scaled_w / orig_w)
        # hot_y = int(orig_hot_y * scaled_h / orig_h)

        # cursor = QCursor(cursor_pixmap, hot_x, hot_y)
        # QApplication.setOverrideCursor(cursor)

    def _load_external_font(self, font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print(f"폰트를 로드하지 못했습니다: {font_path}")
            return ""
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            return font_families[0]
        else:
            print("등록된 폰트 이름을 가져오지 못했습니다.")
            return ""

    def _reset_ui_state(self):
        """홈으로 돌아올 때 '최초 실행 상태'로 되돌리기 (카메라는 유지)."""
        # --- 모드 선택 라벨 초기화 ---
        self._selected_mode_idx = None
        self.selected_mode = None
        # next 버튼 비활성화
        if hasattr(self, "_mode_next_btn") and self._mode_next_btn:
            self._mode_next_btn.setEnabled(False)
        # 라벨 스타일 초기화
        if (
            hasattr(
                self,
                "past_label",
            )
            and self.past_label
        ):
            self.past_label.setProperty("selected", "false")
            self.past_label.style().unpolish(self.past_label)
            self.past_label.style().polish(self.past_label)
            self.past_label.update()
        if hasattr(self, "future_label") and self.future_label:
            self.future_label.setProperty("selected", "false")
            self.future_label.style().unpolish(self.future_label)
            self.future_label.style().polish(self.future_label)
            self.future_label.update()

        # --- 캡처/AI 파이프라인 상태 초기화 (카메라 off 안 함) ---
        self.ai_running = False
        self.poses_left = 0
        self._pose_done_count = 0
        self.candidates = []
        self.final_slots = [None, None]
        self.slot_source = [None, None]
        self.captured_png_bytes = None
        self.captures = []

        # 캡처 페이지 라벨/진행표시 리셋
        if hasattr(self, "lbl_countdown") and self.lbl_countdown:
            self.lbl_countdown.setText("여기를 봐주세요")
        if hasattr(self, "lbl_progress") and self.lbl_progress:
            self.lbl_progress.setText(f"0 / {getattr(self, 'capture_target_count', 1)}")
        if hasattr(self, "btn_next_on_capture") and self.btn_next_on_capture:
            self.btn_next_on_capture.setEnabled(False)

        # --- pick2 페이지 UI 리셋 ---
        if hasattr(self, "sel_labels"):
            for lbl in self.sel_labels:
                if lbl:
                    lbl.clear()
                    lbl.setText("여기에 선택")
                    lbl.setStyleSheet(
                        self._empty_style if hasattr(self, "_empty_style") else ""
                    )
        if hasattr(self, "thumb_labels"):
            for lbl in self.thumb_labels:
                if lbl:
                    lbl.clear()
                    lbl.setEnabled(False)
                    lbl.setStyleSheet(
                        self._thumb_disabled if hasattr(self, "_thumb_disabled") else ""
                    )

        if hasattr(self, "pick2_next_btn") and self.pick2_next_btn:
            self.pick2_next_btn.setEnabled(False)

        # --- 프레임 선택/미리보기 리셋 ---
        self.selected_frame_index = 0
        if hasattr(self, "frame_preview") and self.frame_preview:
            self.frame_preview.clear()
        if hasattr(self, "frame_opt_labels"):
            for i, frame in enumerate(self.frame_opt_labels):
                if frame:
                    # 썸네일 원래대로
                    frame.setStyleSheet(
                        self._frame_thumb_style
                        if hasattr(self, "_frame_thumb_style")
                        else ""
                    )
                    if (
                        i < len(getattr(self, "frame_templates", []))
                        and not self.frame_templates[i].isNull()
                    ):
                        self._set_pix_to_label(frame, self.frame_templates[i])
        self.final_composed_pixmap = QPixmap()

        # --- 인쇄 페이지 미리보기 리셋 ---
        if hasattr(self, "print_preview") and self.print_preview:
            self.print_preview.clear()

        # --- ✅ QR 상태 초기화 추가 ---
        self.qrcode_pixmap = None
        if hasattr(self, "qrcode_label") and self.qrcode_label:
            self.qrcode_label.clear()
            self.qrcode_label.setToolTip("")
        if hasattr(self, "qr"):
            self.qr = QRCODE()  # 새 인스턴스로 교체해서 URL 중복 방지

    def _setup_print_page(self):
        """6번째 인쇄 페이지 초기 설정"""
        self.print_page_index = None
        if self.stacked.count() >= 6:
            self.print_page_index = 5
            page = self.stacked.widget(self.print_page_index)
        else:
            return

        # ui 파일에 아래 두 위젯이 있다고 가정: print_preview(QLabel), btn_print(QPushButton)
        self.print_preview = getattr(page, "print_preview", None)
        self.btn_home = getattr(page, "btn_home", None)
        self.btn_print = getattr(page, "btn_print", None)
        self.qrcode_label = getattr(page, "qrcode_label", None)

        if self.btn_print:
            self.btn_print.clicked.connect(self._print_final_frame)
        if self.btn_home:
            self.btn_home.clicked.connect(lambda: self.goto_page(0))

    def _scale_for_print(self, pm: QPixmap, target_size: QSize, mode: str) -> QPixmap:
        """미리보기/인쇄에서 공통으로 쓰는 스케일 방식"""
        if mode == "stretch":
            # 여백/잘림 없이 꽉 채우되 비율 무시 → 왜곡 가능
            return pm.scaled(target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        elif mode == "fit":
            # 비율 유지 + 종이 안에 맞춤 → 잘림 없음, 여백 가능
            return pm.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # 기존 cover(잘 채우기, 중앙 크롭) – 필요시 유지
            return pm.scaled(
                target_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )

    def _enter_print_page(self):  # 프린터
        """6페이지 들어올 때 미리보기 갱신"""
        if (
            self.print_preview
            and hasattr(self, "final_composed_pixmap")
            and not self.final_composed_pixmap.isNull()
        ):
            self._set_pix_to_label(self.print_preview, self.final_composed_pixmap)

        # ⬇️ 여기 조건 간단히: 라벨만 있으면 생성/표시 시도
        if getattr(self, "qrcode_label", None):
            try:
                # QRCODE 인스턴스가 없다면 만들어두기
                if not hasattr(self, "qr") or self.qr is None:
                    self.qr = QRCODE()

                qr_path, page_url = self.qr.run(self.final_composed_pixmap)

                # 생성된 QR 이미지를 라벨에 표시 (헬퍼 재사용)
                self.qrcode_pixmap = QPixmap(qr_path)
                if not self.qrcode_pixmap.isNull():
                    self._set_pix_to_label(self.qrcode_label, self.qrcode_pixmap)
                    self.qrcode_label.setToolTip(page_url)
                else:
                    print("⚠️ QR 이미지 로드 실패:", qr_path)

            except Exception as e:
                print("QR 생성 실패:", e)

    def _print_final_frame(self):
        if (
            not hasattr(self, "final_composed_pixmap")
        ) or self.final_composed_pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "오류", "출력할 이미지가 없습니다.")
            return

        pm = self.final_composed_pixmap

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.NativeFormat)
        printer.setPrinterName("Canon SELPHY CP1300")  # 기본 프린터 쓰면 주석

        # 100×148mm, 테두리 없음(드라이버에서 Borderless 선택)
        printer.setPaperSize(QSizeF(100, 148), QPrinter.Millimeter)
        printer.setFullPage(True)
        printer.setPageMargins(0, 0, 0, 0, QPrinter.Millimeter)
        printer.setOrientation(QPrinter.Portrait)
        printer.setResolution(300)

        painter = QPainter(printer)
        # === 포인트: 윈도우 좌표를 "이미지 픽셀 크기"로 설정하여 1:1 매핑 ===
        painter.setWindow(0, 0, self.CANVAS_W, self.CANVAS_H)
        painter.drawPixmap(
            0, 0, pm
        )  # 스케일 없이 꽉 채움 (드라이버가 그대로 용지에 맞춤)
        painter.end()

        QtWidgets.QMessageBox.information(
            self, "인쇄", "✅ 프린터로 전송했습니다. 인쇄가 완료되면 가져가세요."
        )

    def _load_frame_boxes(self):
        try:
            if os.path.exists(self._frame_boxes_path):
                with open(self._frame_boxes_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 간단 검증 (프레임 수/박스 수 같을 때만 반영)
                if isinstance(data, list) and all(
                    isinstance(x, list) and len(x) == 2 for x in data
                ):
                    self.frame_boxes_norm = data
        except Exception as e:
            print("[frame_boxes] load failed:", e)

    def _save_frame_boxes(self):
        try:
            with open(self._frame_boxes_path, "w", encoding="utf-8") as f:
                json.dump(self.frame_boxes_norm, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[frame_boxes] save failed:", e)

    def _compose_frame(self, idx: int) -> QPixmap:
        if not (0 <= idx < len(self.frame_templates)):
            return QPixmap()

        base = self.frame_templates[idx]
        if base.isNull() or not all(self.final_slots):
            return QPixmap()

        # 캔버스 생성(최종 출력 크기와 동일)
        canvas = QPixmap(self.CANVAS_W, self.CANVAS_H)
        canvas.fill(Qt.transparent)

        painter = QPainter(canvas)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # 프레임 먼저 그림 (이미 CANVAS_W×CANVAS_H 크기)
        painter.drawPixmap(0, 0, base)

        # 정규화된 박스를 실제 픽셀 좌표로 변환 (캔버스 기준)
        boxes = []
        for nx, ny, nw, nh in self.frame_boxes_norm[idx]:
            x = int(nx * self.CANVAS_W)
            y = int(ny * self.CANVAS_H)
            w = int(nw * self.CANVAS_W)
            h = int(nh * self.CANVAS_H)
            boxes.append(QRect(x, y, max(1, w), max(1, h)))

        # 두 장을 각 박스에 "꽉 채우기" (비율 유지, center-crop)
        for slot_pix, rect in zip([self.final_slots[0], self.final_slots[1]], boxes):
            if isinstance(slot_pix, QPixmap) and not slot_pix.isNull():
                scaled = slot_pix.scaled(
                    rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                x_off = max(0, (scaled.width() - rect.width()) // 2)
                y_off = max(0, (scaled.height() - rect.height()) // 2)
                cropped = scaled.copy(x_off, y_off, rect.width(), rect.height())
                painter.drawPixmap(rect.topLeft(), cropped)

        painter.end()
        return canvas

    def _boxes_from_norm(self, idx: int, base_pix: QPixmap):
        """정규화(0~1) 박스 → 템플릿 실제 픽셀 좌표 QRect 리스트로 변환"""
        if not (0 <= idx < len(self.frame_boxes_norm)):
            return []
        W, H = base_pix.width(), base_pix.height()
        rects = []
        for nx, ny, nw, nh in self.frame_boxes_norm[idx]:
            x = int(nx * W)
            y = int(ny * H)
            w = int(nw * W)
            h = int(nh * H)
            # 테두리 침범 방지 살짝 안쪽으로(선택): 2px 인셋
            inset = 2
            rects.append(
                QRect(
                    x + inset, y + inset, max(1, w - 2 * inset), max(1, h - 2 * inset)
                )
            )
        return rects

    def _init_progress_ui(self):
        if hasattr(self, "progress_dlg") and self.progress_dlg is not None:
            return

        self.progress_dlg = QtWidgets.QDialog()
        self.progress_dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.progress_dlg.setModal(True)
        self.progress_dlg.setObjectName("ai_progress_dialog")

        layout = QtWidgets.QVBoxLayout(self.progress_dlg)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.progress_label = QtWidgets.QLabel("AI 이미지 변환 중 ... 0%")
        self.progress_label.setAlignment(Qt.AlignCenter)
        font = self.progress_label.font()
        font.setPointSize(font.pointSize() + 2)
        self.progress_label.setFont(font)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet(
            """
            QDialog#ai_progress_dialog { background:#111; border:2px solid #4CAF50; border-radius:12px; }
            QProgressBar { background:#222; border:1px solid #333; border-radius:9px; }
            QProgressBar::chunk { background:#4CAF50; border-radius:9px; }
            QLabel { color:#eee; }
        """
        )

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        # 크기 및 중앙 배치
        self.progress_dlg.resize(420, 120)

    def _show_progress(self, text="AI 이미지 변환 중 ...", value=0):
        self._init_progress_ui()
        self.progress_label.setText(f"{text} {int(value)}%")
        self.progress_bar.setValue(int(value))

        geo = self.frameGeometry()
        center = geo.center()

        dlg_geo = self.progress_dlg.frameGeometry()
        dlg_geo.moveCenter(center)
        self.progress_dlg.move(dlg_geo.topLeft())
        self.progress_dlg.show()

        QtWidgets.QApplication.processEvents()

    def _update_progress(self, text, value):
        if hasattr(self, "progress_dlg") and self.progress_dlg.isVisible():
            self.progress_bar.setValue(int(value))
            self.progress_label.setText(f"{text} … {int(value)}%")
            QtWidgets.QApplication.processEvents()

    def _hide_progress(self):
        if hasattr(self, "progress_dlg") and self.progress_dlg.isVisible():
            self.progress_dlg.hide()

    def _setup_capture_page(self):
        self.capture_page_index = None
        self.cap = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._draw_frame)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._tick_countdown)
        self.count_left = 0

        self.captures = []
        self.capture_target_count = 1
        self.selected_mode = None  # 이전 모드에서 저장용

        if self.stacked.count() >= 3:
            self.capture_page_index = 2
            page = self.stacked.widget(self.capture_page_index)

            self.lbl_webcam = getattr(page, "label_webcam", None)
            self.btn_capture = getattr(page, "btn_capture", None)
            self.lbl_countdown = getattr(page, "label_countdown", None)
            self.lbl_progress = getattr(page, "label_progress", None)
            self.btn_next_on_capture = getattr(page, "btn_next", None)
            self.btn_back = getattr(page, "btn_back", None)

            if self.lbl_countdown:
                self.lbl_countdown.setText("여기를 봐주세요")
            if self.lbl_progress:
                self.lbl_progress.setText(f"0 /{self.capture_target_count}")
            if self.btn_next_on_capture:
                self.btn_next_on_capture.setEnabled(False)

            if self.btn_capture:
                self.btn_capture.clicked.connect(self._start_countdown)

    def _enter_capture_page(self):
        self.captures.clear()
        if self.lbl_countdown:
            self.lbl_countdown.setText("여기를 봐주세요")
        if self.lbl_progress:
            self.lbl_progress.setText(f"0 / {self.capture_target_count}")
        if self.btn_next_on_capture:
            self.btn_next_on_capture.setEnabled(False)

        self._start_camera()

    def _start_camera(self):
        if cv2 is None:
            QtWidgets.QMessageBox.critical(
                self, "오류", "OpenCV(cv2)가 설치되어 있지 않습니다."
            )
            return
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(self.camera_port)  ##
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "오류", "카메라를 열 수 없습니다.")
            self.cap.release()
            self.cap = None
            return
        self.video_timer.start(30)  # ~33fps 사진 미리 보기용. 없으면 프레임 멈쳐있음

    def _stop_camera(self):
        self.video_timer.stop()
        self.countdown_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _draw_frame(self):
        if self.cap is None or self.lbl_webcam is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        target = self.lbl_webcam.size()  # 라벨 안쪽 크기
        self.lbl_webcam.setPixmap(
            pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_webcam.setAlignment(Qt.AlignCenter)

        self._last_frame_bgr = frame

    def _start_countdown(self):
        if self.cap is None:
            QtWidgets.QMessageBox.information(
                self, "안내", "카메라가 시작되지 않았습니다."
            )
            return
        if len(self.captures) >= self.capture_target_count:
            QtWidgets.QMessageBox.information(
                self, "안내", "이미 1장을 모두 촬영했습니다."
            )
            return

        self.count_left = 5
        if self.lbl_countdown:
            self.lbl_countdown.setText(str(self.count_left))
        self.countdown_timer.start(1000)

    def _tick_countdown(self):
        self.count_left -= 1
        if self.count_left > 0:
            if self.lbl_countdown:
                self.lbl_countdown.setText(str(self.count_left))
        else:
            self.countdown_timer.stop()

            if self.lbl_countdown:
                self.lbl_countdown.setText("찰칵!")

            if hasattr(self, "_last_frame_bgr") and self._last_frame_bgr is not None:
                success, buf = cv2.imencode(
                    ".png", self._last_frame_bgr
                )  # self._last_frame_bgr로 교체
                if success:
                    self.captured_png_bytes = bytes(buf)
                else:
                    self.captured_png_bytes = None

                self.captures.append(self._last_frame_bgr.copy())

            # 진행표시 업데이트
            if self.lbl_progress:
                self.lbl_progress.setText(
                    f"{len(self.captures)} / {self.capture_target_count}"
                )

            # 0.4초 뒤 카운트 라벨 지우기
            QTimer.singleShot(
                400, lambda: self.lbl_countdown and self.lbl_countdown.setText("")
            )

            # 4장 촬영 완료 시 다음 버튼 활성화
            if (
                len(self.captures) >= self.capture_target_count
                and self.btn_next_on_capture
            ):
                self.btn_next_on_capture.setEnabled(True)

    def _write_mode_buttons(self):
        target = None

        for page in self.pages:
            if hasattr(page, "past_label") and hasattr(page, "future_label"):
                target = page
                self.past_label = page.past_label
                self.future_label = page.future_label
                break

        self._selected_mode_idx = None

        self.past_label.clicked.connect(lambda: self._on_label_mode_clicked(0))
        self.future_label.clicked.connect(lambda: self._on_label_mode_clicked(1))

        target.setStyleSheet(
            """
                /* ✅ 선택된 상태 — 파스텔 레드 강조 */
                QLabel#past_label[selected="true"],
                QLabel#future_label[selected="true"],
                *#past_label[selected="true"],
                *#future_label[selected="true"] {
                    background-color: #E58A8A;     /* 파스텔 레드 */
                    color: #FFFFFF;
                    border: 10px solid #E58A8A;
                    border-radius: 12px;
                }

                /* ✅ 마우스 오버(hover) 시 — 더 밝은 파스텔 레드 테두리 */
                QLabel#past_label:hover,
                QLabel#future_label:hover,
                *#past_label:hover,
                *#future_label:hover {
                    background-color: transparent;
                    color: #E58A8A;                /* 파스텔 레드 */
                    border: 10px solid #EEA4A4;    /* 밝은 파스텔 레드 테두리 */
                    border-radius: 12px;
                }
                """
        )

        btn_next = getattr(target, "btn_next", None)

        if btn_next:
            btn_next.setEnabled(False)
            self._mode_next_btn = btn_next
            self._update_mode_label_styles()

    def _on_label_mode_clicked(self, idx: int):
        # 예전 `_on_mode_chosen(0/1)` 동작을 그대로 사용
        self._selected_mode_idx = idx
        self._on_mode_chosen(idx)  # 기존 코드: self.selected_mode 설정 + next enable
        self._update_mode_label_styles()

    def _update_mode_label_styles(self):
        for i, lbl in enumerate(
            [getattr(self, "past_label", None), getattr(self, "future_label", None)]
        ):
            if not lbl:
                continue
            lbl.setProperty(
                "selected", "true" if i == self._selected_mode_idx else "false"
            )
            # 동적 property 반영
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
            lbl.update()

    def _on_mode_chosen(self, mode_id: int):
        # mode_id == 0(past), mode_id ==1(future)
        self.selected_mode = "past" if mode_id == 0 else "future"
        if hasattr(self, "_mode_next_btn") and self._mode_next_btn:
            self._mode_next_btn.setEnabled(True)

    def _setup_pick2_page(self):
        self.pick2_page_index = None

        if self.stacked.count() >= 4:
            self.pick2_page_index = 3
            page = self.stacked.widget(self.pick2_page_index)
        else:
            return

        self.sel_labels = [getattr(page, "sel_1", None), getattr(page, "sel_2", None)]

        self.thumb_labels = [
            getattr(page, "thumb_1", None),
            getattr(page, "thumb_2", None),
            getattr(page, "thumb_3", None),
        ]

        self.pick2_next_btn = getattr(page, "btn_next", None)

        self.final_slots = [None, None]  # 다시 살리기
        self.slot_source = [None, None]  # 다시 살리기
        self.candidates = []

        self._empty_style = "border: 3px dashed #D0C7B5; background:#F0EEE8; color:#7A6F67; font-size:20px;"  # 눈톤 + 따뜻한 그레이

        self._filled_style = "border: 3px solid #2E6B3F; background:#F9F9F5;"
        # 포레스트 그린 + 밝은 눈색 배경

        self._thumb_style = "border: 2px solid transparent; background:#FAFAF7;"
        # 밝은 화이트 베이지

        self._thumb_disabled = "border: 2px solid #AAA; background:#E1DFDB; opacity:0.6;"  # 흐린 겨울 그레이톤

        for i, lbl in enumerate(self.thumb_labels):
            if lbl:
                lbl.setStyleSheet(self._thumb_style)
                lbl.clicked.connect(lambda t=i: self._choose_from_thumb(t))

        for i, lbl in enumerate(self.sel_labels):
            if lbl:
                lbl.clicked.connect(lambda t=i: self._clear_slot(t))

        if self.pick2_next_btn:
            self.pick2_next_btn.setEnabled(False)

    def _enter_pick2_page(self, pixmaps: list):
        if self.pick2_page_index is None:
            return
        self.candidates = pixmaps[:4]
        for i, lbl in enumerate(self.thumb_labels):
            if not lbl:
                continue
            if i < len(self.candidates) and isinstance(self.candidates[i], QPixmap):
                self._set_pix_to_label(lbl, self.candidates[i])
                lbl.setEnabled(True)
                lbl.setStyleSheet(self._thumb_style)
                lbl.setToolTip("클릭하면 위의 빈 칸에 들어갑니다.")
            else:
                lbl.setPixmap(QPixmap())
                lbl.setText("")
                lbl.setEnabled(False)
                lbl.setStyleSheet(self._thumb_disabled)

        self.final_slots = [None, None]
        self.slot_source = [None, None]
        for lbl in self.sel_labels:
            if lbl:
                lbl.setPixmap(QPixmap())
                lbl.setText("여기에 선택")
                lbl.setStyleSheet(self._empty_style)

        if self.pick2_next_btn:
            self.pick2_next_btn.setEnabled(False)

    def _choose_from_thumb(self, t_index: int):
        """하단 썸네일 클릭 → 다음 빈 슬롯에 채우기"""
        if t_index >= len(self.candidates):
            return
        thumb = self.thumb_labels[t_index]
        if not thumb or not thumb.isEnabled():
            return

        # 빈 슬롯 찾기
        try:
            slot_idx = self.final_slots.index(None)  # None 값인 위치의 인덱스를 반환
        except ValueError:
            # 이미 둘 다 찼으면 무시 (원한다면 마지막 슬롯을 교체하도록 바꿔도 됨)
            return

        pix = self.candidates[t_index]
        self.final_slots[slot_idx] = pix
        self.slot_source[slot_idx] = t_index

        # 슬롯에 그리기
        target_lbl = self.sel_labels[slot_idx]
        if target_lbl:
            self._set_pix_to_label(target_lbl, pix)
            target_lbl.setStyleSheet(self._filled_style)
            target_lbl.setText("")

        # 썸네일 비활성화(중복 선택 방지)
        thumb.setEnabled(False)
        thumb.setStyleSheet(self._thumb_disabled)

        # 둘 다 찼으면 다음 버튼 활성화
        if all(self.final_slots) and self.pick2_next_btn:
            self.pick2_next_btn.setEnabled(True)

    def _set_pix_to_label(self, lbl, pix: QPixmap):
        """라벨 크기에 맞춰 비율 유지로 그림(왜곡 방지)"""
        if not lbl or pix.isNull():
            return
        lbl.setAlignment(Qt.AlignCenter)
        target = lbl.size()
        lbl.setPixmap(pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _clear_slot(self, slot_idx: int):
        if not (0 <= slot_idx < 2):
            return

        pix = self.final_slots[slot_idx]
        if pix is None:
            return

        self.final_slots[slot_idx] = None
        src = self.slot_source[slot_idx]
        self.slot_source[slot_idx] = None

        lbl = self.sel_labels[slot_idx]
        if lbl:
            lbl.setPixmap(QPixmap())
            lbl.setText("여기에 선택")
            lbl.setStyleSheet(self._empty_style)

        if src is not None and 0 <= src < len(self.thumb_labels):
            th = self.thumb_labels[src]
            if th:
                th.setEnabled(True)
                th.setStyleSheet(self._thumb_style)

        if self.pick2_next_btn:
            self.pick2_next_btn.setEnabled(False)

    def _start_ai_pipeline(self):
        if self.ai_running:
            return
        self.ai_running = True

        if hasattr(self, "lbl_countdown") and self.lbl_countdown:
            self.lbl_countdown.setText("AI 변환 작업 중 입니다...")

        self._show_progress("AI 이미지 변환 중…", 0)

        mode = "past" if (self.selected_mode in (None, "past")) else "future"

        if self.pick2_page_index is not None:
            for i, lbl in enumerate(self.thumb_labels):
                if lbl:
                    lbl.clear()
                    lbl.setText("AI 이미지 생성 중...")
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setEnabled(False)
                    lbl.setStyleSheet(self._thumb_disabled)

            self.candidates = [None, None, None, None]

            for lbl in self.sel_labels:
                if lbl:
                    lbl.clear()
                    lbl.setText("여기에 선택")
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setStyleSheet(self._empty_style)
            if self.pick2_next_btn:
                self.pick2_next_btn.setEnabled(False)

        self._age_weight = 25.0
        self._pose_weight_total = 75.0
        self._pose_per = self._pose_weight_total / max(1, len(self.pose_prompts))
        self._pose_done_count = 0

        job = AgeJob(self.captured_png_bytes, mode, token=self.replicate_token, seed=42)
        job.signals.age_done.connect(self._on_age_done)
        job.signals.error.connect(self._on_ai_error)
        self.pool.start(job)

    def _on_ai_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "AI 생성 오류", msg)

    def _on_age_done(self, base_url: str):
        # 🔍 base_url이 URL이면 이미지 저장 (디버그용)
        import requests

        try:
            if isinstance(base_url, str) and base_url.startswith("http"):
                response = requests.get(base_url, timeout=10)
                if response.status_code == 200:
                    with open("my-image.png", "wb") as f:
                        f.write(response.content)
                    print("[DEBUG] Saved base_url image → my-image.png")
                else:
                    print(
                        f"[DEBUG] Failed to download image, status={response.status_code}"
                    )
            else:
                print(f"[DEBUG] base_url is not a valid URL: {base_url}")
        except Exception as e:
            print(f"[DEBUG] Error saving base_url image: {e}")

        inputs = [base_url]
        if hasattr(self, "captured_png_bytes") and self.captured_png_bytes:
            inputs.append(self.captured_png_bytes)

        self.poses_left = len(self.pose_prompts)

        for i, p in enumerate(self.pose_prompts):
            job = PoseJob(
                inputs=inputs,
                pose_prompt=p,
                index=i,
                token=self.replicate_token,
                seed=42,
                aspect_ratio="1:1",
                resolution="720p",
            )
            job.signals.pose_done.connect(self._on_pose_done_bytes)
            job.signals.error.connect(self._on_ai_error)
            QTimer.singleShot(i * 2000, lambda j=job: self.pool.start(j))

    def _on_pose_done_bytes(self, index, data: bytes):
        # 모델 출력(한 장) 그대로 썸네일에 표시
        pm = QPixmap()
        pm.loadFromData(data)

        if 0 <= index < len(self.thumb_labels):
            lbl = self.thumb_labels[index]
            if lbl and not pm.isNull():
                self._set_pix_to_label(lbl, pm)
                lbl.setEnabled(True)
                lbl.setStyleSheet(self._thumb_style)
                lbl.setToolTip("클릭하면 위의 빈 칸에 들어갑니다.")
                if index < len(self.candidates):
                    self.candidates[index] = pm

        self._pose_done_count += 1
        progress = (
            self._age_weight
            + min(self._pose_done_count, len(self.pose_prompts)) * self._pose_per
        )
        self._update_progress("타임머신 완료, 포즈 생성 중", progress)

        self.poses_left -= 1
        if self.poses_left <= 0:
            self.ai_running = False
            self._update_progress("변환이 완료되었습니다.", 100)
            self._hide_progress()
            if self.pick2_page_index is not None:
                self.goto_page(self.pick2_page_index)

    def _setup_frame_page(self):
        self.frame_page_index = None
        if self.stacked.count() >= 5:
            self.frame_page_index = 4
            page = self.stacked.widget(self.frame_page_index)
        else:
            return

        self.frame_preview = getattr(page, "frame_preview", None)
        self.frame_opt_labels = [
            getattr(page, "frame_opt_1", None),
            getattr(page, "frame_opt_2", None),
        ]

        self._frame_thumb_style = """
            QFrame {
                border: 2px solid transparent;
                border-radius: 12px;
                background-color: transparent;
            }
            QFrame:hover {
                border: 2px solid #C94C46; /* 크리스마스 브릭 레드 */
                background-color: rgba(201, 76, 70, 0.15); /* 크리스마스 레드 투명 */
            }
            """

        self._frame_thumb_selected = """
        QFrame {
            border: 2px solid #C94C46;   /* 포인트 레드 */
            border-radius: 12px;
            background-color: #C94C46;   /* 깊은 크리스마스 레드 */
        }
        """

        for frame in self.frame_opt_labels:
            if frame:
                frame.setStyleSheet(self._frame_thumb_style)

        for i, frame in enumerate(self.frame_opt_labels):
            if not frame:
                continue
            frame.setStyleSheet(self._frame_thumb_style)
            if i < len(self.frame_templates) and not self.frame_templates[i].isNull():
                self._set_pix_to_label(frame, self.frame_templates[i])  # ★ 썸네일 표시
            # 클릭 연결 (QLabel이면 mousePressEvent로 대체)
            try:
                frame.clicked.connect(lambda idx=i: self._choose_frame(idx))
            except Exception:
                frame.mousePressEvent = lambda ev, idx=i: self._choose_frame(idx)

        self.selected_frame_index = 0

        self.title_label = getattr(page, "title_label", None)
        if self.title_label:
            self.title_label.clicked.connect(self._open_frame_editor)

        # 5페이지의 next 버튼이 btn_next 라고 가정
        self.frame_next_btn = getattr(page, "btn_next", None)
        if self.frame_next_btn:
            self.frame_next_btn.clicked.connect(
                lambda: (self.goto_page(self.print_page_index),)
            )

    def _open_frame_editor(self):
        idx = self.selected_frame_index
        if idx < 0 or idx >= len(self.frame_templates):
            return
        base = self.frame_templates[idx]
        dlg = FrameEditorDialog(base, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.norms:
            # 현재 프레임의 박스 좌표 교체
            self.frame_boxes_norm[idx] = dlg.norms
            self._save_frame_boxes()
            # 미리보기 즉시 갱신
            self._choose_frame(idx)

    def _choose_frame(self, idx: int):
        self.selected_frame_index = idx
        for i, frame in enumerate(self.frame_opt_labels):
            if not frame:
                continue
            frame.setStyleSheet(
                self._frame_thumb_selected if i == idx else self._frame_thumb_style
            )

        composed = self._compose_frame(idx)  # ← 합성 결과
        if not composed.isNull():
            self.final_composed_pixmap = composed
            if self.frame_preview:
                self._set_pix_to_label(self.frame_preview, composed)

    def goto_page(self, index: int):
        if 0 <= index < self.stacked.count():

            if index == 0:
                self._reset_ui_state()

            if (
                hasattr(self, "capture_page_index")
                and self.capture_page_index is not None
            ):
                if self.stacked.currentIndex() == self.capture_page_index:
                    self._stop_camera()

            self.stacked.setCurrentIndex(index)

            if (
                hasattr(self, "capture_page_index")
                and self.capture_page_index is not None
            ):
                if index == self.capture_page_index:
                    self._enter_capture_page()

            if hasattr(self, "frame_page_index") and index == self.frame_page_index:
                # 저장된 좌표로 미리보기 다시 그리기
                QTimer.singleShot(
                    0, lambda: self._choose_frame(self.selected_frame_index)
                )

            if hasattr(self, "print_page_index") and index == self.print_page_index:
                self._enter_print_page()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.setWindowTitle("타임머신 포토부스")

    # 완전 풀스크린 모드 (타이틀바, 최소화/닫기 버튼 안 보임)
    main_window.showFullScreen()

    sys.exit(app.exec_())
