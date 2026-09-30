import io
import os
import re
import sqlite3
import sys
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import handwriting  # noqa: E402
import process_image  # noqa: E402

handwriting.load_env()

JOB_ROOT = Path("/tmp/autograde-web")
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "app.sqlite"
SECRET_PATH = DATA_DIR / "secret.key"
ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
MAX_SHEETS = 20
DEMO_EMAIL = "demo@autograde.app"
DEMO_PASSWORD = "Demo2026!"
DEMO_NAME = "Nhà tuyển dụng"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not SECRET_PATH.exists():
    SECRET_PATH.write_bytes(os.urandom(32))
app.secret_key = SECRET_PATH.read_bytes()


def safe_stem(name):
    cleaned = secure_filename(name or "")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("._")
    return (cleaned or "phieu")[:80]


def _cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_answer_key(path):
    frame = process_image.load_answer_key(path)
    required = {"Section", "Question", "Sub-question", "Answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "File đáp án thiếu cột: " + ", ".join(sorted(missing)) + ". Cần Section, Question, Sub-question, Answer."
        )
    if frame.empty:
        raise ValueError("File đáp án không có dòng nào.")

    part1 = []
    part2 = {}
    part3 = []
    for _, row in frame.iterrows():
        section = _cell(row["Section"])
        question = int(float(row["Question"]))
        sub = _cell(row["Sub-question"]).lower()
        answer = _cell(row["Answer"])
        if answer == "True":
            shown = "Đúng"
        elif answer == "False":
            shown = "Sai"
        else:
            shown = answer
        if section == "I":
            part1.append({"question": question, "answer": shown})
        elif section == "II":
            part2.setdefault(question, {"question": question, "parts": []})
            part2[question]["parts"].append({"label": sub, "answer": shown})
        elif section == "III":
            part3.append({"question": question, "answer": shown})

    part1.sort(key=lambda item: item["question"])
    part3.sort(key=lambda item: item["question"])
    counts = frame["Section"].astype(str).str.strip().value_counts().to_dict()
    summary_bits = []
    for section, label in (("I", "Phần I"), ("II", "Phần II"), ("III", "Phần III")):
        if section in counts:
            summary_bits.append(f"{label}: {int(counts[section])} dòng")
    return {
        "part1": part1,
        "part2": [part2[key] for key in sorted(part2)],
        "part3": part3,
        "summary": ", ".join(summary_bits) or f"{len(frame)} dòng",
    }


def builtin_answer_key():
    return read_answer_key(process_image.DEFAULT_ANSWER_KEY)


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO users (name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                password_hash = excluded.password_hash
            """,
            (DEMO_NAME, DEMO_EMAIL, generate_password_hash(DEMO_PASSWORD), datetime.now(timezone.utc).isoformat()),
        )


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with db() as conn:
        row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    return {
        "user": current_user(),
        "demo_email": DEMO_EMAIL,
        "demo_password": DEMO_PASSWORD,
    }


def render_page(**kwargs):
    kwargs.setdefault("results", None)
    kwargs.setdefault("error", None)
    kwargs.setdefault("key_source", "Đáp án sẵn có")
    if kwargs.get("key_view") is None and current_user():
        kwargs["key_view"] = builtin_answer_key()
    return render_template("index.html", **kwargs)


def explain_grade_error(exc):
    text = str(exc)
    if "Alignment confidence" in text or "corner markers" in text or "outer corner" in text:
        return "Không căn được phiếu. Ảnh cần thấy đủ 4 ô vuông đen ở góc, không bị che và không bị cắt mất."
    if "bubble" in text.lower() or "grid" in text.lower() or "Could not find" in text:
        return "Đã thấy tờ giấy nhưng không đọc được lưới ô đáp án. Thử ảnh rõ hơn, thấy trọn phiếu."
    return "Không chấm được phiếu này."


def grade_upload(image_path, answer_key_path):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        scores, result = process_image.grade_image(
            image_path,
            answer_key_path=answer_key_path,
            output_path=None,
            aligned_output_path=None,
        )
    return scores, result


@app.route("/")
def index():
    return render_page()


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    name = ""
    email = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(name) < 2:
            error = "Tên cần ít nhất 2 ký tự."
        elif not EMAIL_RE.match(email):
            error = "Email chưa đúng định dạng."
        elif len(password) < 8:
            error = "Mật khẩu cần ít nhất 8 ký tự."
        else:
            try:
                with db() as conn:
                    cursor = conn.execute(
                        "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                        (name, email, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
                    )
                    session["user_id"] = cursor.lastrowid
                return redirect(url_for("index"))
            except sqlite3.IntegrityError:
                error = "Email này đã có tài khoản."
    return render_template(
        "auth.html",
        mode="register",
        heading="Đăng ký",
        lead="Tạo tài khoản để chấm phiếu và xem ảnh vừa tải.",
        error=error,
        name=name,
        email=email,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    email = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with db() as conn:
            row = conn.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["id"]
            return redirect(url_for("index"))
        error = "Email hoặc mật khẩu chưa đúng."
    return render_template(
        "auth.html",
        mode="login",
        heading="Đăng nhập",
        lead="Dùng email và mật khẩu đã đăng ký.",
        error=error,
        email=email,
    )


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/answer-key-sample")
def answer_key_sample():
    return send_file(
        process_image.DEFAULT_ANSWER_KEY,
        as_attachment=True,
        download_name="answer_key.csv",
    )


@app.route("/jobs/<job_id>/<filename>")
def job_file(job_id, filename):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    safe = Path(filename).name
    suffix = Path(safe).suffix.lower()
    if suffix not in IMAGE_TYPES:
        abort(404)
    path = JOB_ROOT / job_id / safe
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=IMAGE_TYPES[suffix])


@app.route("/grade", methods=["POST"])
@login_required
def grade():
    sheets = [item for item in request.files.getlist("sheets") if item and item.filename]
    key_file = request.files.get("answer_key")
    if not sheets:
        return render_page(error="Chọn ít nhất một ảnh phiếu."), 400
    if len(sheets) > MAX_SHEETS:
        return render_page(error=f"Chỉ chấm tối đa {MAX_SHEETS} phiếu một lần."), 400

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if key_file and key_file.filename:
        suffix = Path(key_file.filename).suffix.lower()
        if suffix != ".csv":
            return render_page(error="Đáp án phải là file CSV."), 400
        answer_key_path = job_dir / "answer_key.csv"
        key_file.save(answer_key_path)
        key_source = key_file.filename
    else:
        answer_key_path = process_image.DEFAULT_ANSWER_KEY
        key_source = "Đáp án sẵn có"

    try:
        key_view = read_answer_key(answer_key_path)
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else "Không đọc được file đáp án."
        return render_page(error=message), 400

    results = []
    for index, sheet in enumerate(sheets, start=1):
        original = sheet.filename
        suffix = Path(original).suffix.lower()
        entry = {"name": original, "ok": False}
        if suffix not in ALLOWED_IMAGES:
            entry["error"] = "Chỉ nhận ảnh JPG, PNG, WEBP, BMP hoặc TIFF."
            results.append(entry)
            continue
        stored = job_dir / f"{index:02d}_{safe_stem(original)}"
        if stored.suffix.lower() not in ALLOWED_IMAGES:
            stored = stored.with_suffix(suffix or ".jpg")
        sheet.save(stored)
        try:
            scores, image = grade_upload(stored, answer_key_path)
        except Exception as exc:
            traceback.print_exc()
            entry["error"] = explain_grade_error(exc)
            results.append(entry)
            continue
        graded_name = f"{index:02d}_graded.jpg"
        graded_path = job_dir / graded_name
        cv2.imwrite(str(graded_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        entry.update({
            "ok": True,
            "original": f"/jobs/{job_id}/{stored.name}",
            "image": f"/jobs/{job_id}/{graded_name}",
            "part1": scores["part1"],
            "part2": scores["part2"],
            "part3": scores["part3"],
            "student_id": scores["student_id"],
            "exam_code": scores["exam_code"],
            "student_name": scores.get("student_name", ""),
            "birth_date": scores.get("birth_date", ""),
            "room": scores.get("room", ""),
            "school": scores.get("school", ""),
            "council": scores.get("council", ""),
            "subject": scores.get("subject", ""),
            "exam_name": scores.get("exam_name", ""),
            "exam_date": scores.get("exam_date", ""),
            "identity_note": scores.get("identity_note", ""),
        })
        results.append(entry)

    return render_page(results=results, key_view=key_view, key_source=key_source)


if __name__ == "__main__":
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    init_db()
    app.run(host="127.0.0.1", port=5050, debug=False)
