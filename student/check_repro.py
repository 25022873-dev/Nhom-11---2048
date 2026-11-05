"""Kiểm tra tái lập: chạy cùng một seed nhiều lần và báo có giống nhau không.

Chạy từ thư mục gốc project:

    py -m student.check_repro                     # seed 42, 3 lần, tổng budget 60 giây
    py -m student.check_repro --seed 2000 --runs 5

Nếu các lần chạy lệch nhau, công cụ nói rõ nguyên nhân (ván bị cắt vì hết budget, hay đồng
hồ chen vào quyết định) và gợi ý cách chỉnh. Đây chỉ là công cụ phụ: ``agent.py`` KHÔNG
import file này.
"""

from __future__ import annotations

import argparse
import hashlib

from game2048.benchmark import run_game

from student.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy một seed nhiều lần và so sánh kết quả")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--total-sec", type=float, default=60.0)
    args = parser.parse_args()

    print(f"Tốc độ máy dùng cho quỹ node: {Agent._nodes_per_sec():,.0f} node/giây "
          f"(nguồn: student/speed.json hoặc tự đo)")
    rows = []
    for index in range(args.runs):
        agent = Agent()
        result = run_game(agent, args.seed, None, False, args.total_sec)
        board = hashlib.md5(str(result["final_board"]).encode()).hexdigest()[:8]
        rows.append({
            "score": result["score"], "moves": result["moves"], "board": board,
            "seconds": result["total_decision_time_ms"] / 1000.0,
            "exhausted": bool(result["budget_exhausted"]), "clock": agent.clock_events,
            "bad": result["illegal_moves"] + result["exceptions"],
        })
        row = rows[-1]
        print(f"  lần {index + 1}: điểm {row['score']:>7,} | nước {row['moves']:>5} | bàn cờ cuối {row['board']} | "
              f"dùng {row['seconds']:5.1f}s/{args.total_sec:g}s | hết budget: {'CÓ' if row['exhausted'] else 'không'} | "
              f"đồng hồ chen vào: {row['clock']} | lỗi: {row['bad']}")

    identical = len({(r["score"], r["moves"], r["board"]) for r in rows}) == 1
    print()
    if identical and not any(r["clock"] or r["exhausted"] for r in rows):
        print(f"KẾT LUẬN: TÁI LẬP ĐƯỢC. {args.runs} lần chạy giống hệt nhau và đồng hồ không chen vào quyết định nào.")
        return
    if identical:
        print("KẾT LUẬN: các lần chạy giống nhau, nhưng đồng hồ đã chạm tới giới hạn (xem bên dưới) nên có rủi ro lệch.")
    else:
        print("KẾT LUẬN: KHÔNG TÁI LẬP. Các lần chạy khác nhau.")
    if any(r["exhausted"] for r in rows):
        print("- Ván bị cắt vì hết tổng budget: máy chậm hơn mức agent đã đo, hoặc đang bận. "
              "Xóa student/speed.json để đo lại khi máy rảnh, hoặc giảm Agent.POOL_FRACTION (ví dụ 0.45).")
    if any(r["clock"] for r in rows):
        print("- Đồng hồ đã chen vào quyết định (lúc sắp hết budget). Cách xử lý giống mục trên.")
    if not any(r["exhausted"] or r["clock"] for r in rows) and not identical:
        print("- Không thấy đồng hồ can thiệp nhưng kết quả vẫn lệch: hãy gửi nguyên văn kết quả này để kiểm tra tiếp.")


if __name__ == "__main__":
    main()
