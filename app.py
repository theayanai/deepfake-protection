from __future__ import annotations

import json
import os
import re
import shutil
import textwrap
import time
import urllib.request
import uuid
from pathlib import Path
from string import Template
from typing import Any

import cv2
import gradio as gr
import google.generativeai as genai
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import pipeline
import importlib


load_dotenv()

APP_NAME = "Deepfake Protection AI"
DEFAULT_CELEBRITY_NAME = "Reference Identity"
DEEPFAKE_THRESHOLD = 0.8
IDENTITY_THRESHOLD = 0.75
REFERENCE_IMAGE_FALLBACK = Path("data/celebrity/srk.jpg")
TEST_DIR = Path("data/test")
CELEBRITY_DIR = Path("data/celebrity")
UI_DIR = Path("ui")

TEST_DIR.mkdir(parents=True, exist_ok=True)
CELEBRITY_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
        print(message, flush=True)


def clean_text(value: Any) -> str:
        if value is None:
                return ""
        return str(value).strip()


def bool_to_badge(flag: bool) -> str:
        return "TRUE" if flag else "FALSE"


def safe_float(value: Any, default: float = 0.0) -> float:
        try:
                return float(value)
        except Exception:
                return default


def safe_int(value: Any, default: int = 0) -> int:
        try:
                return int(value)
        except Exception:
                return default


def safe_json_loads(text: str) -> dict[str, Any]:
        if not text:
                return {}
        try:
                return json.loads(text)
        except Exception:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                        try:
                                return json.loads(match.group(0))
                        except Exception:
                                return {}
        return {}


def ensure_rgb_image_file(image_path: str | Path | None) -> str | None:
        if not image_path:
                return None
        image_path = str(image_path)
        if not os.path.exists(image_path):
                return None
        return image_path


def save_numpy_image(image: np.ndarray, output_path: Path) -> Path:
        if image is None:
                raise ValueError("No image data supplied.")

        if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)
        return output_path


def save_file_copy(source_path: str | Path, output_path: Path) -> Path:
        source_path = str(source_path)
        if not os.path.exists(source_path):
                raise FileNotFoundError(source_path)
        shutil.copy2(source_path, output_path)
        return output_path


def extract_face(image_path: str | Path) -> tuple[bool, int, tuple[int, int, int, int] | None]:
        image = cv2.imread(str(image_path))
        if image is None:
                return False, 0, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) == 0:
                return False, 0, None

        x, y, w, h = faces[0]
        return True, len(faces), (int(x), int(y), int(w), int(h))


def extract_video_frame(video_path: str | Path, output_path: Path, second: float = 1.0) -> Path | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
                return None

        try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

                if fps > 0:
                        frame_index = safe_int(min(total_frames - 1, max(0, fps * second)), 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                ok, frame = cap.read()
                if not ok or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()

                if not ok or frame is None:
                        return None

                cv2.imwrite(str(output_path), frame)
                return output_path
        finally:
                cap.release()


def fetch_youtube_thumbnail(youtube_url: str, output_path: Path) -> dict[str, Any]:
        if not youtube_url:
                raise ValueError("Paste a YouTube URL first.")

        try:
                import yt_dlp
        except Exception as exc:
                raise RuntimeError(f"yt-dlp is unavailable: {exc}") from exc

        ydl_opts = {"quiet": True, "skip_download": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

        thumbnail_url = info.get("thumbnail")
        if not thumbnail_url:
                thumbnails = info.get("thumbnails") or []
                if thumbnails:
                        thumbnail_url = thumbnails[-1].get("url")

        if not thumbnail_url:
                raise RuntimeError("Could not resolve a thumbnail for this YouTube link.")

        urllib.request.urlretrieve(thumbnail_url, output_path)

        return {
                "title": info.get("title", "Unknown Title"),
                "channel": info.get("uploader", info.get("channel", "Unknown Channel")),
                "webpage_url": info.get("webpage_url", youtube_url),
                "thumbnail_url": thumbnail_url,
                "duration": info.get("duration"),
        }


def load_gemini_model() -> genai.GenerativeModel | None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
                log("GEMINI_API_KEY not found. Gemini features will fall back to local heuristics.")
                return None

        try:
                genai.configure(api_key=api_key)
                return genai.GenerativeModel("gemini-1.5-flash")
        except Exception as exc:
                log(f"Gemini initialization failed: {exc}")
                return None


def load_deepfake_detector():
        log("Loading deepfake detector...")
        device = 0 if torch.cuda.is_available() else -1
        try:
                return pipeline(
                        "image-classification",
                        model="dima806/deepfake_vs_real_image_detection",
                        device=device,
                )
        except Exception as exc:
                log(f"Deepfake model fallback engaged: {exc}")
                return None


def load_deepface_class():
        try:
                module = importlib.import_module("deepface")
                return getattr(module, "DeepFace", None)
        except Exception as exc:
                log(f"DeepFace import fallback engaged: {exc}")
                return None


GEMINI_MODEL = load_gemini_model()
DEEPFAKE_DETECTOR = load_deepfake_detector()
DEEPFACE_CLASS = load_deepface_class()


def clear_cuda_cache() -> None:
        if torch.cuda.is_available():
                torch.cuda.empty_cache()


def verify_identity(test_img_path: str, reference_img_path: str) -> dict[str, Any]:
        if DEEPFACE_CLASS is None:
                return {"verified": False, "distance": 1.0, "threshold": IDENTITY_THRESHOLD}

        try:
                result = DEEPFACE_CLASS.verify(
                        img1_path=test_img_path,
                        img2_path=reference_img_path,
                        enforce_detection=False,
                        distance_metric="cosine",
                        model_name="Facenet512",
                )
                verified = bool(result.get("verified", False))
                distance = safe_float(result.get("distance"), 1.0)
                threshold = safe_float(result.get("threshold"), IDENTITY_THRESHOLD)
                clear_cuda_cache()
                return {
                        "verified": verified,
                        "distance": distance,
                        "threshold": threshold,
                }
        except Exception as exc:
                log(f"Identity verification error: {exc}")
                clear_cuda_cache()
                return {"verified": False, "distance": 1.0, "threshold": IDENTITY_THRESHOLD}


def get_deepfake_score(image_path: str) -> dict[str, Any]:
        if DEEPFAKE_DETECTOR is None:
                return {"score": 0.5, "label": "unknown", "source": "fallback"}

        try:
                results = DEEPFAKE_DETECTOR(image_path)
                if not results:
                        return {"score": 0.5, "label": "unknown", "source": "empty"}

                best = results[0]
                label = str(best.get("label", "unknown"))
                score = safe_float(best.get("score"), 0.5)

                if label.lower() == "fake":
                        fake_score = score
                elif label.lower() == "real":
                        fake_score = 1.0 - score
                else:
                        fake_score = score

                clear_cuda_cache()
                return {
                        "score": round(max(0.0, min(fake_score, 1.0)), 4),
                        "label": label,
                        "source": "model",
                }
        except Exception as exc:
                log(f"Deepfake model error: {exc}")
                clear_cuda_cache()
                return {"score": 0.62, "label": "fallback", "source": "error"}


def build_local_decision(celebrity_name: str, deepfake_score: float, identity_match: bool, face_detected: bool) -> dict[str, Any]:
        risk = (deepfake_score * 0.66) + (0.28 if identity_match else 0.0) + (0.06 if face_detected else 0.0)
        risk = max(0.0, min(risk, 1.0))
        misuse = bool(identity_match and deepfake_score >= DEEPFAKE_THRESHOLD)
        if not face_detected:
                misuse = False
                risk = min(risk, 0.35)

        return {
                "misuse": misuse,
                "confidence": round(risk, 3),
                "reason": (
                        f"Local heuristic flagged the content because the face was{' ' if face_detected else ' not '}detected, "
                        f"identity match was {'positive' if identity_match else 'negative'}, and the deepfake score was {deepfake_score:.3f}."
                ),
                "celebrity": celebrity_name,
                "risk_score": round(risk, 3),
                "recommended_action": "Draft takedown and escalate to platform trust & safety." if misuse else "No takedown recommended from local heuristic.",
        }


def analyze_case(
        celebrity_name: str,
        source_label: str,
        source_details: str,
        deepfake_score: float,
        identity_result: dict[str, Any],
        face_detected: bool,
) -> tuple[dict[str, Any], str]:
        local_json = build_local_decision(
                celebrity_name=celebrity_name,
                deepfake_score=deepfake_score,
                identity_match=identity_result["verified"],
                face_detected=face_detected,
        )

        if GEMINI_MODEL is None:
                return local_json, json.dumps(local_json, indent=2)

        prompt = textwrap.dedent(
                f"""
                You are a defensive deepfake-review assistant for a hackathon demo.

                Goal:
                - Review the scan and output one JSON object only.
                - Decide if the content is likely misuse or a malicious fake.
                - Keep the response short and factual.

                Context:
                - Reference identity: {celebrity_name}
                - Source: {source_label}
                - Source details: {source_details}
                - Face detected: {face_detected}
                - Identity verified: {identity_result['verified']}
                - Identity distance: {identity_result['distance']:.4f}
                - Deepfake score: {deepfake_score:.4f}

                Return JSON with exactly these keys:
                {{
                    "misuse": true/false,
                    "confidence": 0 to 1,
                    "reason": "short explanation",
                    "recommended_action": "short action phrase"
                }}
                """
        ).strip()

        try:
                generation_config = getattr(genai, "GenerationConfig", None)
                kwargs: dict[str, Any] = {"temperature": 0.2}
                if generation_config is not None:
                        kwargs["generation_config"] = generation_config(response_mime_type="application/json")
                response = GEMINI_MODEL.generate_content(prompt, **kwargs)
                raw_text = clean_text(getattr(response, "text", ""))
                parsed = safe_json_loads(raw_text)
                if not parsed:
                        parsed = local_json
                        raw_text = json.dumps(local_json, indent=2)

                parsed.setdefault("misuse", local_json["misuse"])
                parsed.setdefault("confidence", local_json["confidence"])
                parsed.setdefault("reason", local_json["reason"])
                parsed.setdefault("recommended_action", local_json["recommended_action"])
                return parsed, raw_text
        except Exception as exc:
                fallback = dict(local_json)
                fallback["reason"] = f"Gemini fallback: {exc}. {local_json['reason']}"
                return fallback, json.dumps(fallback, indent=2)


def generate_notice(celebrity_name: str, analysis_json: dict[str, Any], source_label: str, source_details: str) -> str:
        if GEMINI_MODEL is None:
                return textwrap.dedent(
                        f"""
                        Subject: Takedown Request for Suspected Misleading Content

                        Dear Platform Trust & Safety Team,

                        I am requesting review and removal of content that appears to impersonate or misuse the identity of {celebrity_name}.

                        Source: {source_label}
                        Details: {source_details}
                        Gemini review: {json.dumps(analysis_json, indent=2)}

                        Please remove or restrict the content pending review.

                        Sincerely,
                        Deepfake Protection AI
                        """
                ).strip()

        prompt = textwrap.dedent(
                f"""
                Draft a professional DMCA-style takedown notice for a defensive hackathon demo.

                Constraints:
                - Keep it formal, concise, and ready to send.
                - Do not add legal disclaimers.
                - Mention this is based on AI-assisted review of suspected misuse.

                Inputs:
                - Target identity: {celebrity_name}
                - Source: {source_label}
                - Source details: {source_details}
                - Analysis JSON: {json.dumps(analysis_json, indent=2)}
                """
        ).strip()

        try:
                response = GEMINI_MODEL.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.25))
                return clean_text(getattr(response, "text", "")) or "Gemini notice generation returned no text."
        except Exception as exc:
                return f"Gemini notice generation failed: {exc}"


def resolve_reference_image(reference_image: str | None) -> str | None:
        if reference_image:
                resolved = ensure_rgb_image_file(reference_image)
                if resolved:
                        return resolved
        if REFERENCE_IMAGE_FALLBACK.exists():
                return str(REFERENCE_IMAGE_FALLBACK)
        return None


def resolve_suspicious_frame(
        source_mode: str,
        uploaded_image: str | None,
        uploaded_video: str | None,
        youtube_url: str | None,
) -> tuple[str | None, str, str, dict[str, Any]]:
        run_id = uuid.uuid4().hex
        source_details = ""
        source_label = source_mode
        metadata: dict[str, Any] = {}

        if source_mode == "Image Upload":
                path = ensure_rgb_image_file(uploaded_image)
                if not path:
                        return None, source_label, "Upload an image to begin.", metadata
                output_path = TEST_DIR / f"scan_{run_id}.jpg"
                save_file_copy(path, output_path)
                source_details = f"Local image: {Path(path).name}"
                return str(output_path), source_label, source_details, metadata

        if source_mode == "Local Video":
                path = ensure_rgb_image_file(uploaded_video)
                if not path:
                        return None, source_label, "Upload a video to extract a frame.", metadata
                output_path = TEST_DIR / f"frame_{run_id}.jpg"
                frame_path = extract_video_frame(path, output_path)
                if frame_path is None:
                        return None, source_label, "Could not extract a frame from the uploaded video.", metadata
                source_details = f"Local video frame: {Path(path).name}"
                return str(frame_path), source_label, source_details, metadata

        if source_mode == "YouTube URL":
                if not youtube_url:
                        return None, source_label, "Paste a YouTube URL to fetch the preview frame.", metadata
                output_path = TEST_DIR / f"youtube_{run_id}.jpg"
                try:
                        metadata = fetch_youtube_thumbnail(youtube_url, output_path)
                except Exception as exc:
                        return None, source_label, f"YouTube fetch failed: {exc}", metadata
                source_details = metadata.get("title", youtube_url)
                return str(output_path), source_label, source_details, metadata

        return None, source_label, "Unknown source mode.", metadata


def build_dashboard_html(
        celebrity_name: str,
        source_label: str,
        source_details: str,
        face_detected: bool,
        face_count: int,
        identity_result: dict[str, Any],
        deepfake_result: dict[str, Any],
        analysis_json: dict[str, Any],
        elapsed_seconds: float,
) -> str:
        risk = safe_float(analysis_json.get("confidence"), 0.0)
        risk_width = max(0.0, min(risk * 100, 100))
        verdict = "HIGH RISK" if analysis_json.get("misuse") else "LOW RISK"
        badge_class = "danger" if analysis_json.get("misuse") else "safe"
        badge_text = "Potential misuse detected" if analysis_json.get("misuse") else "No strong misuse signal"

        def card(label: str, value: str, hint: str = "") -> str:
                return f"""                <div class='metric-card'>
                    <div class='metric-label'>{clean_text(label)}</div>
                    <div class='metric-value'>{clean_text(value)}</div>
                    <div class='metric-hint'>{clean_text(hint)}</div>
                </div>
                """

        metric_cards_html = "\n".join(
                [
                        card("Reference", celebrity_name, source_details),
                        card("Face Scan", "Detected" if face_detected else "Missing", f"Faces seen: {face_count}"),
                        card(
                                "Identity Match",
                                "YES" if identity_result["verified"] else "NO",
                                f"Distance: {identity_result['distance']:.4f} | Threshold: {identity_result['threshold']:.4f}",
                        ),
                        card("Deepfake Score", f"{deepfake_result['score']:.3f}", f"Classifier: {deepfake_result['label']}"),
                ]
        )

        dashboard_template = load_ui_text("dashboard.html")
        if not dashboard_template.strip():
                return "<div class='error-banner'>Missing UI template: ui/dashboard.html</div>"

        return Template(dashboard_template).safe_substitute(
                verdict=verdict,
                badge_text=badge_text,
                celebrity_name=clean_text(celebrity_name),
                source_label=clean_text(source_label),
                badge_class=badge_class,
                metric_cards_html=metric_cards_html,
                risk_width=f"{risk_width:.2f}",
                risk_value=f"{risk:.3f}",
                elapsed_seconds=f"{elapsed_seconds:.2f}s",
                face_detected_badge=bool_to_badge(face_detected),
                model_source=clean_text(deepfake_result.get("source", "unknown")),
                analysis_json_pretty=json.dumps(analysis_json, indent=2),
        )


def scan_content(
        source_mode: str,
        uploaded_image: str | None,
        uploaded_video: str | None,
        youtube_url: str | None,
        reference_image: str | None,
        celebrity_name: str,
) -> tuple[str | None, str, dict[str, Any], str, str]:
        started = time.perf_counter()
        celebrity_name = clean_text(celebrity_name) or DEFAULT_CELEBRITY_NAME
        reference_path = resolve_reference_image(reference_image)

        if reference_path is None:
                message = "Upload a reference celebrity image before running identity matching."
                return None, f"<div class='error-banner'>{message}</div>", {}, "", message

        try:
                suspicious_path, resolved_source, source_details, metadata = resolve_suspicious_frame(
                        source_mode=source_mode,
                        uploaded_image=uploaded_image,
                        uploaded_video=uploaded_video,
                        youtube_url=youtube_url,
                )
        except Exception as exc:
                message = f"Failed to resolve input source: {exc}"
                return None, f"<div class='error-banner'>{message}</div>", {}, "", message

        if suspicious_path is None:
                message = source_details
                return None, f"<div class='error-banner'>{message}</div>", {}, "", message

        face_detected, face_count, face_box = extract_face(suspicious_path)
        identity_result = verify_identity(suspicious_path, reference_path)
        deepfake_result = get_deepfake_score(suspicious_path)
        analysis_json, raw_analysis_text = analyze_case(
                celebrity_name=celebrity_name,
                source_label=resolved_source,
                source_details=source_details,
                deepfake_score=deepfake_result["score"],
                identity_result=identity_result,
                face_detected=face_detected,
        )

        if "confidence" not in analysis_json:
                analysis_json["confidence"] = round((deepfake_result["score"] * 0.66) + (0.28 if identity_result["verified"] else 0.0), 3)

        notice = generate_notice(
                celebrity_name=celebrity_name,
                analysis_json=analysis_json,
                source_label=resolved_source,
                source_details=source_details,
        )

        elapsed_seconds = time.perf_counter() - started
        dashboard_html = build_dashboard_html(
                celebrity_name=celebrity_name,
                source_label=resolved_source,
                source_details=source_details,
                face_detected=face_detected,
                face_count=face_count,
                identity_result=identity_result,
                deepfake_result=deepfake_result,
                analysis_json=analysis_json,
                elapsed_seconds=elapsed_seconds,
        )

        status_line = [
                f"Source mode: {resolved_source}",
                f"File: {Path(suspicious_path).name}",
                f"Face detected: {face_detected} (count={face_count})",
                f"Face box: {face_box}",
                f"Identity verified: {identity_result['verified']} | distance={identity_result['distance']:.4f} | threshold={identity_result['threshold']:.4f}",
                f"Deepfake score: {deepfake_result['score']:.4f} | label={deepfake_result['label']}",
                f"Gemini source: {'online' if GEMINI_MODEL is not None else 'fallback'}",
        ]
        if metadata:
                status_line.append(f"YouTube metadata: {json.dumps(metadata, indent=2)}")
        status_line.append(f"Raw Gemini response: {raw_analysis_text}")

        return suspicious_path, dashboard_html, analysis_json, notice, "\n".join(status_line)


def toggle_source_inputs(source_mode: str):
        return (
                gr.update(visible=source_mode == "Image Upload"),
                gr.update(visible=source_mode == "Local Video"),
                gr.update(visible=source_mode == "YouTube URL"),
        )


def load_ui_text(filename: str, fallback: str = "") -> str:
                path = UI_DIR / filename
                try:
                                return path.read_text(encoding="utf-8")
                except Exception as exc:
                                log(f"UI asset fallback for {path}: {exc}")
                                return fallback


APP_CSS = load_ui_text("styles.css")
APP_JS = load_ui_text("app.js")


def hero_html() -> str:
                return load_ui_text("hero.html")


def source_help_text() -> str:
                return load_ui_text("source_help.html")


if __name__ == "__main__":
        with gr.Blocks(theme="huggingface", css=APP_CSS, js=APP_JS, title=APP_NAME) as demo:
                gr.HTML(hero_html())
                gr.HTML(source_help_text())

                with gr.Row():
                        with gr.Column(scale=5):
                                with gr.Group(elem_classes=["glass-panel", "control-panel"]):
                                        gr.Markdown("## Scan Studio")
                                        gr.Markdown("Choose a source and tune the detection run before you launch the scan.")

                                        source_mode = gr.Radio(
                                                ["Image Upload", "Local Video", "YouTube URL"],
                                                value="Image Upload",
                                                label="Evidence Source",
                                        )

                                        uploaded_image = gr.Image(
                                                type="filepath",
                                                label="Suspicious Image Upload",
                                                visible=True,
                                        )

                                        uploaded_video = gr.Video(
                                                type="filepath",
                                                label="Suspicious Video Upload",
                                                visible=False,
                                        )

                                        youtube_url = gr.Textbox(
                                                label="YouTube URL",
                                                placeholder="https://www.youtube.com/watch?v=...",
                                                visible=False,
                                        )

                                        reference_image = gr.Image(
                                                type="filepath",
                                                label="Reference Identity Image",
                                                info="Upload a trusted image for the person you are checking.",
                                        )

                                        celebrity_name = gr.Textbox(
                                                label="Reference Identity Name",
                                                value=DEFAULT_CELEBRITY_NAME,
                                                placeholder="Name of the person being checked",
                                        )

                                        run_button = gr.Button("Launch Deepfake Sweep", variant="primary")

                        with gr.Column(scale=7):
                                evidence_image = gr.Image(label="Extracted Evidence Frame", type="filepath")
                                dashboard_html = gr.HTML()
                                analysis_json = gr.JSON(label="Gemini Decision JSON")
                                dmca_notice = gr.Textbox(label="Draft DMCA Notice", lines=16)
                                scan_log = gr.Textbox(label="Evidence Trace", lines=10)

                gr.HTML("<div class='footer-mark'>Built for hackathon-grade demo impact</div>")

                source_mode.change(
                        toggle_source_inputs,
                        inputs=[source_mode],
                        outputs=[uploaded_image, uploaded_video, youtube_url],
                )

                run_button.click(
                        scan_content,
                        inputs=[source_mode, uploaded_image, uploaded_video, youtube_url, reference_image, celebrity_name],
                        outputs=[evidence_image, dashboard_html, analysis_json, dmca_notice, scan_log],
                )

        demo.launch(debug=True)