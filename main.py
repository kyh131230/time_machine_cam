import sys, os, glob, cv2
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QRunnable, QThreadPool, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter
from setting import FileController
from replicate_tasks import AgeJob, PoseJob
import numpy as np


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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ai_running = False
        self.poses_left = 0

        self.stacked = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked)

        # ui/*.ui 를 알파벳 순서로 자동 로드 (first.ui, second.ui, ...)
        self.pages = []
        self.captured_png_bytes = None

        self.frame_template_paths = [
            resource_path("frame_1.png"),
            resource_path("frame_1.png"),
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
            [(0.10, 0.07, 0.80, 0.40), (0.10, 0.53, 0.80, 0.40)],
            # frame_2
            [(0.08, 0.05, 0.84, 0.42), (0.08, 0.53, 0.84, 0.42)],
        ]

        self.goto_page(0)  # 첫 화면
        self._write_mode_buttons()

        self._setup_capture_page()
        self._setup_pick2_page()
        self._setup_frame_page()

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

        TWO_PERSON_LOCK = (
            "Use both people from the input photo together in one frame. "
            "Show exactly two people. No single-person composition."
        )

        POSE_PROMPTS = [
            " Both people stand side by side, smile, and give a thumbs-up.",
            " Both people smile and take a selfie with a phone.",
            " Both people smile and make a heart shape with their hands.",
        ]
        self.pose_prompts = POSE_PROMPTS

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
                    ".png", cv2.imread("qwer_2.jpg")
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
        self._update_progress(self._age_weight)

        inputs = [base_url]
        if hasattr(self, "captured_png_bytes") and self.captured_png_bytes:
            inputs.append(self.captured_png_bytes)

        self.poses_left = len(self.pose_prompts)

        for i, p in enumerate(self.pose_prompts):
            job = PoseJob(inputs, p, index=i, token=self.replicate_token, seed=42)
            job.signals.pose_done.connect(self._on_pose_done_bytes)
            job.signals.error.connect(self._on_ai_error)
            self.pool.start(job)

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

    def _choose_frame(self, idx: int):
        self.selected_frame_index = idx
        for i, frame in enumerate(self.frame_opt_labels):
            if not frame:
                continue
            frame.setStyleSheet(
                self._frame_thumb_selected if i == idx else self._frame_thumb_style
            )

        composed = self._compose_frame(idx)  # ← 합성 결과
        if self.frame_preview and not composed.isNull():
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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.setWindowTitle("타임머신 포토부스")

    # 완전 풀스크린 모드 (타이틀바, 최소화/닫기 버튼 안 보임)
    main_window.showFullScreen()

    sys.exit(app.exec_())
