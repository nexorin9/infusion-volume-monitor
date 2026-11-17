#!/usr/bin/env python3
"""把一份**真实终端会话记录**渲染成 PNG 截图。

    python scripts/render_terminal_shot.py --in var/shotcapture/01-upload.txt \\
        --out docs/screenshots/01-upload-and-curves.png --title "上传帧序列 · 面积曲线与毫升"

渲染的是**已经真实跑过的命令的 stdout/stderr 文本**（由
``scripts/capture_demo_shots.py`` 落盘的会话记录），不是手写的示意图，也不是
后期拼出来的效果图：文本一个字都没改，只是排版成终端的样子。

字体走系统字体栈：ASCII 用 Menlo（等宽），中文用 Hiragino Sans GB，按
东亚全角宽度占两格，所以终端里的对齐关系在图上依然成立。

配色是深色终端主题（背景 ``#11141a``），文字层级只有四档：命令、注释、
普通输出、状态码高亮。**没有**任何示意性插画——概念图在
``docs/concepts/``，实测截图在 ``docs/screenshots/``，两者不混用。
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: JSON 键名（引号包住、后面紧跟冒号）
_JSON_KEY_RE = re.compile(r'"[^"]*"(?=\s*:)')

# ── 配色（深色终端主题，四档文字层级） ─────────────────────────────────
BG = "#11141a"
TITLEBAR = "#1b1f27"
CHROME = "#2c313a"
FG = "#d7dae0"
FG_DIM = "#7d8590"
FG_CMD = "#f2f4f8"
FG_PROMPT = "#5fb27a"
FG_OK = "#5fb27a"
FG_WARN = "#d9a441"
FG_ERR = "#e05561"
FG_ACCENT = "#63a2d9"
FG_JSON_KEY = "#9dc4e8"

MONO_PATH = "/System/Library/Fonts/Menlo.ttc"
CJK_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
MONO_INDEX = 0
CJK_INDEX = 0


def _is_wide(char: str) -> bool:
    return unicodedata.east_asian_width(char) in ("W", "F")


class ShotRenderer:
    """把会话文本逐行画到一张 PNG 上（等宽网格，中文占两格）。"""

    def __init__(self, size: int = 15, wrap: int = 118) -> None:
        self.size = size
        self.wrap = wrap
        self.mono = ImageFont.truetype(MONO_PATH, size, index=MONO_INDEX)
        self.cjk = ImageFont.truetype(CJK_PATH, size, index=CJK_INDEX)
        self.advance = self.mono.getlength("M")
        self.line_height = int(size * 1.55)
        self.pad = 18
        self.title_height = 34

    def _font_for(self, char: str) -> ImageFont.FreeTypeFont:
        return self.cjk if _is_wide(char) else self.mono

    @staticmethod
    def _display_width(text: str) -> int:
        return sum(2 if _is_wide(char) else 1 for char in text)

    def _wrap(self, line: str) -> list[str]:
        """按终端宽度折行。

        命令行按词折、续行以 ``\\`` 结尾（粘回终端仍可执行）；输出行按字符硬折，
        与真实终端的行为一致。中文按两格计宽。
        """
        if self._display_width(line) <= self.wrap:
            return [line]
        if line.startswith("$ "):
            out: list[str] = []
            current = ""
            for word in line.split(" "):
                candidate = f"{current} {word}" if current else word
                if current and self._display_width(candidate) > self.wrap:
                    out.append(current + " \\")
                    current = "    " + word
                else:
                    current = candidate
            out.append(current)
            return out
        chunks: list[str] = []
        current = ""
        width = 0
        for char in line:
            char_width = 2 if _is_wide(char) else 1
            if width + char_width > self.wrap:
                chunks.append(current)
                current, width = "", 0
            current += char
            width += char_width
        if current:
            chunks.append(current)
        return chunks


    def _line_style(self, line: str) -> tuple[str, bool]:
        """返回 (颜色, 是否加粗)。只做四档极简判定，不做花哨语法高亮。"""
        stripped = line.strip()
        if not stripped:
            return FG, False
        if stripped.startswith("$ ") or stripped.startswith("# "):
            return FG_CMD, False
        if stripped.startswith("HTTP/") or stripped.startswith("← ") or stripped.startswith("→ "):
            if " 5" in stripped[:12] or stripped.startswith("HTTP/1.1 4"):
                return FG_ERR, False
            return FG_OK, False
        if stripped.startswith("!"):
            return FG_WARN, False
        if stripped.startswith("#"):
            return FG_DIM, False
        if stripped.startswith(('"', "{")) or stripped.startswith("]"):
            return FG, False
        return FG, False

    def _segments(self, line: str) -> list[tuple[str, str]]:
        """把一行拆成 (文本, 颜色) 段：JSON 键名高亮，其余沿用行级颜色。"""
        base = self._line_style(line)[0]
        segments: list[tuple[str, str]] = []
        pos = 0
        for match in _JSON_KEY_RE.finditer(line):
            if match.start() > pos:
                segments.append((line[pos:match.start()], base))
            segments.append((match.group(0), FG_JSON_KEY))
            pos = match.end()
        if pos < len(line):
            segments.append((line[pos:], base))
        return segments or [(line, base)]

    def render(self, text: str, out_path: Path, title: str) -> Path:
        raw_lines = text.replace("\t", "    ").split("\n")
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()
        lines: list[str] = []
        for line in raw_lines:
            lines.extend(self._wrap(line))

        columns = 0
        for line in lines:
            columns = max(columns, self._display_width(line))
        columns = max(columns, len(title) + 12, 40)

        width = int(self.pad * 2 + self.advance * columns) + 2
        height = self.title_height + self.pad * 2 + self.line_height * len(lines)

        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)

        # 标题栏（三个圆点 + 窗口标题），让画面一眼就是终端而不是一张表格
        draw.rectangle([0, 0, width, self.title_height], fill=TITLEBAR)
        for index, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            cx = 18 + index * 18
            draw.ellipse([cx - 6, self.title_height // 2 - 6, cx + 6, self.title_height // 2 + 6], fill=color)
        title_font = ImageFont.truetype(MONO_PATH, max(11, self.size - 3), index=MONO_INDEX)
        title_width = sum(
            title_font.getlength(c) * (2 if _is_wide(c) else 1) for c in title
        )
        tx = width // 2 - int(title_width) // 2
        for char in title:
            font = self._font_for(char)
            draw.text((tx, 8), char, font=font, fill=FG_DIM)
            tx += title_font.getlength(char) * (2 if _is_wide(char) else 1)

        y = self.title_height + self.pad
        for line in lines:
            x = self.pad
            for segment_text, color in self._segments(line):
                for char in segment_text:
                    font = self._font_for(char)
                    draw.text((x, y), char, font=font, fill=color)
                    x += self.advance * (2 if _is_wide(char) else 1)
            y += self.line_height

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")
        return out_path

    @staticmethod
    def _title_font_width(text: str, font: ImageFont.FreeTypeFont) -> int:
        return sum(font.getlength(c) for c in text)


def main() -> int:
    parser = argparse.ArgumentParser(description="把真实终端会话记录渲染成 PNG 截图")
    parser.add_argument("--in", dest="source", required=True, help="会话记录文本文件")
    parser.add_argument("--out", required=True, help="输出 PNG 路径")
    parser.add_argument("--title", default="", help="终端窗口标题")
    parser.add_argument("--size", type=int, default=15, help="字号")
    args = parser.parse_args()

    source = Path(args.source)
    out_path = Path(args.out)
    renderer = ShotRenderer(size=args.size)
    renderer.render(source.read_text(encoding="utf-8"), out_path, args.title or source.name)
    print(f"{out_path} <- {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
