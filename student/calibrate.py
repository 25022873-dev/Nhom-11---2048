"""Đo lại tốc độ máy này và ghi ``student/speed.json`` (công cụ phụ, không bắt buộc).

Agent tự làm việc này ở lần chạy đầu tiên nếu chưa có ``speed.json``. Chạy file này khi
muốn đo lại (ví dụ đổi máy hoặc máy đang bận lúc đo lần đầu), tốt nhất khi máy rảnh:

    py -m student.calibrate

``agent.py`` KHÔNG import file này.
"""

from __future__ import annotations

from student.agent import Agent


def main() -> None:
    print("Đang đo tốc độ (vài giây, hãy để máy rảnh)...")
    value = Agent.calibrate(write=True)
    print(f"nodes_per_sec = {value:,.0f}  (đã ghi student/speed.json nếu thư mục cho phép ghi)")


if __name__ == "__main__":
    main()
