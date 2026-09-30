"""Read the handwritten header of an aligned answer sheet.

The call goes to Gemini through whatever base URL is configured, so a deploy
can use an API gateway without shipping a local OCR model. Bubble fields stay
with the grid reader.
"""

import json
import os
import re
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

HEADER_FIELDS = (
    "subject",
    "exam_name",
    "exam_date",
    "council",
    "school",
    "room",
    "student_name",
    "birth_date",
)
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://api.yescale.io"

_PROMPT = """Đây là phần đầu một phiếu trả lời trắc nghiệm tiếng Việt, đã căn thẳng.
Đọc chữ viết tay trên các dòng kẻ. Trả về JSON với đúng các khóa:
subject, exam_name, exam_date, council, school, room, student_name, birth_date.
Ô trống hoặc không đọc được thì để chuỗi rỗng. Giữ dấu tiếng Việt. Không suy đoán.
Không điền số báo danh hay mã đề. Các ô tròn bên phải không phải họ tên.
"""


def load_env(path=ENV_PATH):
    """Fill missing variables from the project .env. A real environment wins."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _blank(note=""):
    fields = {name: "" for name in HEADER_FIELDS}
    fields["identity_note"] = note
    return fields


def header_crop(image, layout):
    """The lined blanks sit above the first answer row."""
    height = image.shape[0]
    try:
        top = min(point[1] for block in layout["part1"] for row in block for point in row)
        pitch = float(layout.get("pitch_y") or 16)
        y1 = int(round(top - pitch * 1.2))
    except (KeyError, TypeError, ValueError):
        y1 = int(height * 0.36)
    y1 = max(80, min(y1, height))
    return image[:y1, :]


def _parse_header(text):
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("header JSON was not an object")
    cleaned = {}
    for name in HEADER_FIELDS:
        value = data.get(name, "")
        if value is None:
            value = ""
        value = re.sub(r"\s+", " ", str(value)).strip()
        value = re.sub(r"\s*/\s*", "/", value)
        cleaned[name] = value[:80]
    cleaned["identity_note"] = ""
    return cleaned


def read_sheet_header(image, layout):
    """Return the handwritten header, or blank fields with a short note."""
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _blank("Chưa có GEMINI_API_KEY nên chưa đọc phần chữ viết.")

    crop = header_crop(image, layout)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return _blank("Không cắt được phần đầu phiếu.")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _blank("Máy chủ chưa cài thư viện đọc chữ.")

    base_url = (
        os.environ.get("GEMINI_BASE_URL")
        or os.environ.get("base_url")
        or DEFAULT_BASE_URL
    ).strip()
    model = (os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta", base_url=base_url, timeout=40000),
        )
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"),
                _PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        text = getattr(response, "text", "") or ""
        return _parse_header(text)
    except json.JSONDecodeError:
        return _blank("Đọc được ảnh nhưng không tách được các mục chữ.")
    except Exception:
        return _blank("Không đọc được phần chữ viết. Điểm trắc nghiệm vẫn được chấm.")
