import sys, os, glob, cv2
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QRunnable, QThreadPool, QRect
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap, QPainter
from setting import FileController
from replicate_tasks import AgeJob, PoseJob
import numpy as np
import json
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtCore import QSizeF


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

        self.label = QtWidgets.QLabel()
        self.label.setPixmap(base_pixmap)
        self.label.setAlignment(Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        # 상태
        self.rects = []  # 완료된 사각형들(최대 2개)
        self.master_rect = None  # 첫 박스
        self.start_pos = None
        self.drag_pos = None
        self.lock_xw = True  # ✅ 두 번째 박스의 x/width 고정
        self.equal_height = False  # ⏲️ 필요 시 높이까지 동일화

        self.norms = None

        # 이벤트 바인딩
        self.label.mousePressEvent = self._on_mouse_press
        self.label.mouseMoveEvent = self._on_mouse_move
        self.label.mouseReleaseEvent = self._on_mouse_release

        # 도움말
        QtWidgets.QToolTip.showText(
            self.mapToGlobal(self.rect().center()),
            "드래그해서 위/아래 박스 2개를 그리세요.\n"
            "L: 좌우 고정 토글  |  H: 높이 동일 토글",
            self,
        )

    def _on_mouse_move(self, ev):
        if self.start_pos is None:
            return
        self.drag_pos = ev.pos()

        preview = self.base_pixmap.copy()
        p = QPainter(preview)
        p.setPen(Qt.red)

        r = self._current_preview_rect()
        if r:
            p.drawRect(r)

        for rr in self.rects:
            p.drawRect(rr)
        p.end()
        self.label.setPixmap(preview)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_L:
            self.lock_xw = not self.lock_xw
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                f"좌우 고정: {'ON' if self.lock_xw else 'OFF'}",
                self,
            )
        elif e.key() == Qt.Key_H:
            self.equal_height = not self.equal_height
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                f"높이 동일: {'ON' if self.equal_height else 'OFF'}",
                self,
            )
        else:
            super().keyPressEvent(e)

    def _on_mouse_press(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        if len(self.rects) >= 2:
            # 두 개 완료되면 바로 정규화 출력
            self._emit_norm_and_close()
            return
        self.start_pos = ev.pos()
        self.drag_pos = ev.pos()

    def _current_preview_rect(self):
        if self.start_pos is None or self.drag_pos is None:
            return None
        r = QtCore.QRect(self.start_pos, self.drag_pos).normalized()

        # 두 번째 박스부터는 x, width, height 전부 고정
        if self.master_rect:
            r.setX(self.master_rect.x())
            r.setWidth(self.master_rect.width())
            r.setHeight(self.master_rect.height())
        return r

    def _on_mouse_release(self, ev):
        if self.start_pos is None:
            return
        self.drag_pos = ev.pos()
        r = self._current_preview_rect()
        self.start_pos = None
        self.drag_pos = None
        if not r or r.width() <= 0 or r.height() <= 0:
            self.label.setPixmap(self.base_pixmap)
            return

        self.rects.append(r)
        if len(self.rects) == 1:
            self.master_rect = r  # ✅ 첫 박스 기준
        elif len(self.rects) >= 2:
            # 두 번째 박스 높이 동일 강제
            r.setHeight(self.master_rect.height())
            # 두 개 모두 정렬 후 정규화 출력
            self.rects.sort(key=lambda rr: rr.y())
            self._emit_norm_and_close()
            return

        # 갱신
        preview = self.base_pixmap.copy()
        p = QPainter(preview)
        p.setPen(Qt.red)
        for rr in self.rects:
            p.drawRect(rr)
        p.end()
        self.label.setPixmap(preview)

    def _on_mouse_release(self, ev):
        if self.start_pos is None:
            return
        self.drag_pos = ev.pos()
        r = self._current_preview_rect()
        self.start_pos = None
        self.drag_pos = None
        if not r or r.width() <= 0 or r.height() <= 0:
            # 무효 드래그
            self.label.setPixmap(self.base_pixmap)
            return

        self.rects.append(r)
        if len(self.rects) == 1:
            self.master_rect = r  # ✅ 첫 박스를 마스터로 저장
        elif len(self.rects) >= 2:
            # 두 개 모두 그려졌으면 정렬(위→아래) 후 출력
            self.rects.sort(key=lambda rr: rr.y())
            self._emit_norm_and_close()
            return

        # 갱신
        preview = self.base_pixmap.copy()
        p = QPainter(preview)
        p.setPen(Qt.red)
        for rr in self.rects:
            p.drawRect(rr)
        p.end()
        self.label.setPixmap(preview)

    def _emit_norm_and_close(self):
        W, H = self.base_pixmap.width(), self.base_pixmap.height()
        norms = []
        for rr in self.rects[:2]:
            nx = rr.x() / W
            ny = rr.y() / H
            nw = rr.width() / W
            nh = rr.height() / H
            norms.append((round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)))

        self.norms = norms  # ⬅ 결과 보관
        QtWidgets.QApplication.clipboard().setText(str(norms))
        print("✅ frame_boxes_norm:", norms)
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self._frame_boxes_path = os.path.join(
            os.path.dirname(__file__), "frame_boxes.json"
        )

        self.ai_running = False
        self.poses_left = 0

        self.stacked = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked)

        # ui/*.ui 를 알파벳 순서로 자동 로드 (first.ui, second.ui, ...)
        self.pages = []
        self.captured_png_bytes = None

        self.frame_template_paths = [
            resource_path("frame_1.png"),
            resource_path("frame_2.png"),
        ]
        self.frame_templates = [QPixmap(p) for p in self.frame_template_paths]

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

        self.replicate_token = (
            FileController().load_json().get("REPLICATE_API_TOKEN", "")
        )
        if self.replicate_token:
            os.environ["REPLICATE_API_TOKEN"] = self.replicate_token

        POSE_PROMPTS = [
            "Use @orig and @aged as two distinct people standing side by side. Both give a thumbs-up with their right hands. Keep each person’s facial identity, hairstyle, clothing vibe, and age consistent with their own reference. Medium shot, straight-on, 1:1 framing, natural indoor lighting. Do not merge faces; keep @orig and @aged clearly separate.",
            "A realistic selfie composition: @orig holds a smartphone naturally in her right hand and slightly raises it at an upward angle. @aged sits or stands close on the left side, gently leaning toward @orig while both smile and look at the phone screen together. Keep their facial identities, hairstyles, and ages exactly as in the references. Show natural wrist angle and correct phone orientation (no twisted hand). Medium-close shot, 1:1 framing, soft pink background lighting similar to a beauty studio, realistic phone reflection and glow.",
            "Both @orig and @aged face the camera and form a heart shape together with their hands at chest height. Warm, soft light; medium shot, 1:1 composition. Preserve each identity, hairstyle, clothing vibe, and age from references. Keep them as two distinct people—no merging.",
        ]

        self.pose_prompts = POSE_PROMPTS

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
        self.btn_print = getattr(page, "btn_print", None)
        if self.btn_print:
            self.btn_print.clicked.connect(self._print_final_frame)

    def _enter_print_page(self):
        """6페이지 들어올 때 미리보기 갱신"""
        if self.print_preview and not self.final_composed_pixmap.isNull():
            self._set_pix_to_label(self.print_preview, self.final_composed_pixmap)

    def _print_final_frame(self):
        """버튼 클릭 시 바로 포토프린터로 여백 없이 인쇄"""
        if (
            not hasattr(self, "final_composed_pixmap")
            or self.final_composed_pixmap.isNull()
        ):
            QtWidgets.QMessageBox.warning(self, "오류", "출력할 이미지가 없습니다.")
            return

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(
            QPrinter.NativeFormat
        )  # 실제 프린터 출력 => NativeFormat
        printer.setPrinterName("ALPDF")  # (선택) 특정 프린터 지정

        # 10x15cm 용지 + 여백 0(borderless)
        printer.setPaperSize(QSizeF(100, 150), QPrinter.Millimeter)
        printer.setFullPage(True)
        printer.setPageMargins(0, 0, 0, 0, QPrinter.Millimeter)
        printer.setOrientation(QPrinter.Portrait)
        printer.setResolution(300)

        painter = QPainter(printer)
        page = painter.viewport()
        pm = self.final_composed_pixmap
        scaled = pm.scaled(page.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (page.width() - scaled.width()) // 2
        y = (page.height() - scaled.height()) // 2

        painter.drawPixmap(x, y, scaled)
        painter.end()

        QtWidgets.QMessageBox.information(
            self, "인쇄 완료", "✅ 인생네컷 사진이 바로 출력되었습니다!"
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

        # 정규화 박스를 실제 QRect로 변환
        boxes = self._boxes_from_norm(idx, base)

        canvas = QPixmap(base.size())
        canvas.fill(Qt.transparent)

        painter = QPainter(canvas)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # 프레임 그리기
        painter.drawPixmap(0, 0, base)

        # 두 장을 각 박스에 채워 넣기 (비율 유지, 박스 꽉 채우기)
        slots = [self.final_slots[0], self.final_slots[1]]

        for slot_pix, rect in zip(slots, boxes):
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

    def _update_progress(self, value):
        if hasattr(self, "progress_dlg") and self.progress_dlg.isVisible():
            self.progress_bar.setValue(int(value))
            self.progress_label.setText(f"AI 이미지 변환 중… {int(value)}%")
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
        self.cap = cv2.VideoCapture(0)
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

        self.count_left = 1
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
                    ".png", cv2.imread("senior(male).png")
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
            if hasattr(page, "btn_past") and hasattr(page, "btn_future"):
                target = page
                self.btn_past = page.btn_past
                self.btn_future = page.btn_future
                break

        self.btn_past.setCheckable(True)
        self.btn_future.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_past, 0)
        self.mode_group.addButton(self.btn_future, 1)

        target.setStyleSheet(
            """
            QPushButton#btn_past:checked,
            QPushButton#btn_future:checked {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #388E3C;
            }
        """
        )

        self.mode_group.buttonClicked[int].connect(self._on_mode_chosen)

        btn_next = getattr(target, "btn_next", None)
        if btn_next:
            btn_next.setEnabled(False)
            self._mode_next_btn = btn_next

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

        self._empty_style = (
            "border: 3px dashed #bbb; background:#111; color:#999; font-size:20px;"
        )
        self._filled_style = "border: 3px solid #4CAF50; background:#000;"
        self._thumb_style = "border: 2px solid transparent; background:#000;"
        self._thumb_disabled = "border: 2px solid #999; background:#333; opacity:0.6;"

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
        self._update_progress(progress)

        self.poses_left -= 1
        if self.poses_left <= 0:
            self.ai_running = False
            self._update_progress(100)
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

        self._frame_thumb_style = "border: 2px solid transparent; background:#000;"
        self._frame_thumb_selected = "border: 2px solid #4CAF50; background:#000;"

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
                lambda: (
                    self.goto_page(self.print_page_index),
                    self._enter_print_page(),
                )
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
