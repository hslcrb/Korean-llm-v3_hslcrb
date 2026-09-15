import queue
from pathlib import Path
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Frame, Label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

from src.models.korean_llm import KoreanLLM
from src.generation.generator import generate
from src.utils.checkpoint import find_latest_checkpoint
from src.utils.logging_utils import loss_history


class TrainingMonitorGUI:
    def __init__(self, tokenizer, device: torch.device, model_config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_config = model_config
        self.chat_model: Optional[KoreanLLM] = None
        self.current_ckpt_path: Optional[str] = None
        self.running = True
        self.msg_queue: queue.Queue = queue.Queue()

        self.root = tk.Tk()
        self.root.title("KoreanLLM Training Monitor 📊 + Chat")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 좌측 프레임: 손실 곡선 그래프
        left_frame = Frame(self.root, width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        Label(left_frame, text="📉 Loss Curve (실시간)", font=("Arial", 12, "bold")).pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, label="Loss")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 우측 프레임: 채팅창
        right_frame = Frame(self.root, width=550)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ckpt_label = Label(right_frame, text="현재 체크포인트: (아직 없음)", font=("Arial", 10), fg="blue")
        self.ckpt_label.pack(pady=5)

        Label(right_frame, text="💬 모델과 대화하기 (CPU 구동)", font=("Arial", 12, "bold")).pack()

        self.chat_display = scrolledtext.ScrolledText(right_frame, height=25, width=60, state='disabled', wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)

        input_frame = Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = Entry(input_frame, font=("Arial", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)

        send_btn = Button(input_frame, text="전송", command=self.send_message, width=8)
        send_btn.pack(side=tk.RIGHT)

        refresh_btn = Button(right_frame, text="🔄 최신 체크포인트 로드", command=self.load_latest_checkpoint)
        refresh_btn.pack(pady=5)

        self.root.after(1000, self.update_gui)

    def on_close(self):
        self.running = False
        self.root.destroy()

    def update_gui(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg["type"] == "loss":
                    self._update_plot()
                elif msg["type"] == "ckpt":
                    self.current_ckpt_path = msg["path"]
                    self.ckpt_label.config(text=f"현재 체크포인트: {Path(msg['path']).name}")
                elif msg["type"] == "log":
                    self._append_chat(f"[시스템] {msg['text']}\n", "system")
        except queue.Empty:
            pass

        if self.running:
            self.root.after(1000, self.update_gui)

    def _update_plot(self):
        if not loss_history:
            return
        steps = [h["step"] for h in loss_history]
        losses = [h["loss"] for h in loss_history]
        self.line.set_data(steps, losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _append_chat(self, text: str, tag: str = "user"):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def load_latest_checkpoint(self):
        ckpt = find_latest_checkpoint()
        if not ckpt:
            self._append_chat("[시스템] 체크포인트가 아직 없습니다.\n", "system")
            return

        try:
            self._append_chat(f"[시스템] 체크포인트 로딩 중(CPU): {Path(ckpt).name} ...\n", "system")
            self.root.update()

            if self.chat_model is None:
                self.chat_model = KoreanLLM(**self.model_config).to(self.device)

            checkpoint = torch.load(ckpt, map_location=self.device)
            self.chat_model.load_state_dict(checkpoint['model_state_dict'])
            self.chat_model.eval()

            self.current_ckpt_path = ckpt
            self.ckpt_label.config(text=f"현재 체크포인트: {Path(ckpt).name}")
            self._append_chat(f"[시스템] 로드 완료(CPU)! 이제 대화할 수 있어요.\n", "system")
        except Exception as e:
            self._append_chat(f"[시스템] 로드 실패: {e}\n", "system")

    def send_message(self, event=None):
        prompt = self.user_input.get().strip()
        if not prompt:
            return

        self.user_input.delete(0, tk.END)
        self._append_chat(f"나: {prompt}\n", "user")

        if self.chat_model is None:
            self._append_chat("[시스템] 먼저 '최신 체크포인트 로드' 버튼을 눌러주세요.\n", "system")
            return

        try:
            self._append_chat("모델(CPU): 생각 중...\n", "model")
            self.root.update()

            response = generate(
                self.chat_model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=512,
                temperature=0.6,
                top_p=0.95,
                device=self.device
            )

            self.chat_display.config(state='normal')
            self.chat_display.delete("end-2l", "end-1l")
            self.chat_display.config(state='disabled')

            self._append_chat(f"모델: {response}\n\n", "model")
        except Exception as e:
            self._append_chat(f"[시스템] 생성 오류: {e}\n", "system")

    def notify_loss(self):
        self.msg_queue.put({"type": "loss"})

    def notify_checkpoint(self, path: str):
        self.msg_queue.put({"type": "ckpt", "path": path})

    def notify_log(self, text: str):
        self.msg_queue.put({"type": "log", "text": text})

    def run(self):
        self.root.mainloop()
