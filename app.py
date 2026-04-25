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
import librosa
import numpy as np
from PIL import Image
from openai import OpenAI
import torch
from dotenv import load_dotenv
from transformers import pipeline
try:
        from insightface.app import FaceAnalysis  # type: ignore[reportMissingImports]
except Exception:
        FaceAnalysis = None
try:
        from retinaface import RetinaFace  # type: ignore[reportMissingImports]
except Exception:
        RetinaFace = None


load_dotenv()

APP_NAME = "Deepfake Protection AI"
DEFAULT_CELEBRITY_NAME = "Reference Identity"
DEEPFAKE_THRESHOLD = 0.8
IDENTITY_THRESHOLD = 0.75
GEMINI_KEYS = [key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(",") if key.strip()]
CURRENT_KEY_INDEX = 0
DEEPFAKE_MODEL_CANDIDATES = [
        os.getenv("DEEPFAKE_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-Model"),
        "dima806/deepfake_vs_real_image_detection",
]
AUDIO_DEEPFAKE_MODEL_ID = os.getenv("AUDIO_DEEPFAKE_MODEL_ID", "").strip()
REFERENCE_IMAGE_FALLBACK = Path("data/celebrity/srk.jpg")
TEST_DIR = Path("data/test")
CELEBRITY_DIR = Path("data/celebrity")
UI_DIR = Path("ui")

TEST_DIR.mkdir(parents=True, exist_ok=True)
CELEBRITY_DIR.mkdir(parents=True, exist_ok=True)

face_app = None
if FaceAnalysis is not None:
        try:
                face_app = FaceAnalysis(name="buffalo_l")
                try:
                        face_app.prepare(ctx_id=0)
                except Exception:
                        face_app.prepare(ctx_id=-1)
        except Exception as exc:
                print(f"InsightFace initialization fallback engaged: {exc}", flush=True)
                face_app = None

def log(message: str) -> None:
        print(message, flush=True)


def clean_text(value: Any) -> str:
        if value is None:
                return ""
        return str(value).strip()


OPENAI_CLIENT = None
openai_api_key = clean_text(os.getenv("OPENAI_API_KEY"))
if openai_api_key:
        try:
                OPENAI_CLIENT = OpenAI(api_key=openai_api_key)
        except Exception as exc:
                print(f"OpenAI initialization failed: {exc}", flush=True)


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


def ensure_media_file(media_value: Any) -> str | None:
        if media_value is None:
                return None

        if isinstance(media_value, (str, Path)):
                return ensure_rgb_image_file(media_value)

        if isinstance(media_value, dict):
                for key in ("path", "video", "name"):
                        candidate = media_value.get(key)
                        resolved = ensure_rgb_image_file(candidate)
                        if resolved:
                                return resolved

        return None


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
        img = cv2.imread(str(image_path))
        if img is None:
                return False, 0, None

        # 1) InsightFace
        if face_app is not None:
                faces = face_app.get(img)
                if len(faces) > 0:
                        box = faces[0].bbox.astype(int)
                        return True, len(faces), tuple(box)

        # 2) RetinaFace
        if RetinaFace is not None:
                try:
                        detections = RetinaFace.detect_faces(img)
                        if isinstance(detections, dict) and len(detections) > 0:
                                first_key = list(detections.keys())[0]
                                box = detections[first_key]["facial_area"]
                                return True, len(detections), tuple(box)
                except Exception:
                        pass

        # 3) OpenCV Haar cascade fallback
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = haar.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
                x, y, w, h = faces[0]
                return True, len(faces), (int(x), int(y), int(x + w), int(y + h))

        return False, 0, None


def align_face(image_path: str, box: tuple[int, int, int, int] | None) -> str:
        img = cv2.imread(image_path)

        if img is None or box is None:
                return image_path

        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
                return image_path

        face = img[y1:y2, x1:x2]

        if face.size == 0:
                return image_path

        face = cv2.resize(face, (224, 224))

        aligned_path = TEST_DIR / f"aligned_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(aligned_path), face)

        return str(aligned_path)


def draw_face_box(image_path: str, box: tuple[int, int, int, int] | None) -> str:
        img = cv2.imread(image_path)

        if img is None or box is None:
                return image_path

        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
                return image_path

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_path = TEST_DIR / f"boxed_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(out_path), img)

        return str(out_path)


def crop_face(image_path: str, box: tuple[int, int, int, int] | None) -> str | None:
        img = cv2.imread(image_path)

        if img is None or box is None:
                return None

        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
                return None

        face = img[y1:y2, x1:x2]
        if face.size == 0:
                return None

        out_path = TEST_DIR / f"crop_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(out_path), face)
        return str(out_path)


def analyze_face_attributes(image_path: str) -> dict[str, Any]:
        try:
                img = cv2.imread(image_path)

                if img is None or face_app is None:
                        return {}

                faces = face_app.get(img)
                if len(faces) == 0:
                        return {}

                face = faces[0]

                return {
                        "gender": getattr(face, "gender", None),
                        "age": getattr(face, "age", None),
                }
        except Exception:
                return {}


def generate_human_reason(
        identity_result: dict[str, Any],
        ref_attr: dict[str, Any],
        sus_attr: dict[str, Any],
        vision_score: float,
        df_score: float,
) -> str:
        reasons: list[str] = []

        sim = safe_float(identity_result.get("similarity", 0.0), 0.0)
        if sim < 0.3:
                reasons.append("Reference and detected face appear to be different individuals")
        elif sim < 0.5:
                reasons.append("Face similarity is low, identity uncertain")
        else:
                reasons.append("Faces appear similar")

        ref_gender = ref_attr.get("gender")
        sus_gender = sus_attr.get("gender")
        if ref_gender is not None and sus_gender is not None and ref_gender != sus_gender:
                reasons.append("Possible gender mismatch between reference and detected face")

        if vision_score > 0.6:
                reasons.append("Facial features show strong signs of AI-generated content")
        elif df_score > 0.6:
                reasons.append("Visual artifacts suggest potential deepfake manipulation")

        if not reasons:
                reasons.append("No major inconsistencies detected")

        return ". ".join(reasons)


def extract_multiple_frames(video_path: str | Path, num_frames: int = 5) -> list[str]:
        cap = cv2.VideoCapture(str(video_path))
        frames: list[str] = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
                cap.release()
                return frames

        indices = np.linspace(0, total - 1, num_frames).astype(int)

        for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ret, frame = cap.read()
                if ret:
                        path = TEST_DIR / f"frame_{uuid.uuid4().hex}.jpg"
                        cv2.imwrite(str(path), frame)
                        frames.append(str(path))

        cap.release()
        return frames


def audio_score(video_path: str) -> float:
        try:
                y, sr = librosa.load(video_path, sr=16000)
                if y is None or len(y) == 0:
                        return 0.0

                # Spectral feature stack (stronger than raw energy only).
                rms = float(np.mean(librosa.feature.rms(y=y)))
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
                flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
                rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) / (sr / 2))
                centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) / (sr / 2))
                mfcc_var = float(np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)))

                rms_norm = max(0.0, min(rms / 0.12, 1.0))
                zcr_norm = max(0.0, min(zcr / 0.25, 1.0))
                flat_norm = max(0.0, min(flatness / 0.5, 1.0))
                roll_norm = max(0.0, min(rolloff, 1.0))
                cent_norm = max(0.0, min(centroid, 1.0))
                mfcc_norm = max(0.0, min(mfcc_var / 900.0, 1.0))

                feature_score = (
                        0.24 * flat_norm
                        + 0.18 * zcr_norm
                        + 0.18 * roll_norm
                        + 0.15 * cent_norm
                        + 0.15 * mfcc_norm
                        + 0.10 * rms_norm
                )

                model_score = None
                if AUDIO_DETECTOR is not None:
                        try:
                                model_results = AUDIO_DETECTOR(video_path)
                                model_score = parse_audio_model_score(model_results)
                                print("Audio model output:", model_results, flush=True)
                        except Exception as exc:
                                log(f"Audio model error: {exc}")

                if model_score is None:
                        return round(max(0.0, min(feature_score, 1.0)), 4)

                combined = (0.7 * model_score) + (0.3 * feature_score)
                return round(max(0.0, min(combined, 1.0)), 4)
        except Exception:
                return 0.0





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


def get_next_gemini_model() -> genai.GenerativeModel | None:
        global CURRENT_KEY_INDEX

        if not GEMINI_KEYS:
                return None

        key = GEMINI_KEYS[CURRENT_KEY_INDEX % len(GEMINI_KEYS)]
        CURRENT_KEY_INDEX += 1

        try:
                genai.configure(api_key=key)
                return genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
                return None


def call_gemini_with_retry(prompt: str) -> str | None:
        if not GEMINI_KEYS:
                return None

        for _ in range(len(GEMINI_KEYS)):
                model = get_next_gemini_model()
                if model is None:
                        continue
                try:
                        response = model.generate_content(prompt)
                        return clean_text(getattr(response, "text", ""))
                except Exception as exc:
                        print("Gemini failed, trying next key:", exc, flush=True)

        return None


def call_openai(prompt: str) -> str | None:
        if OPENAI_CLIENT is None:
                return None

        try:
                response = OPENAI_CLIENT.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content if response.choices else ""
                return clean_text(content)
        except Exception as exc:
                print("OpenAI failed:", exc, flush=True)
                return None


def load_deepfake_detectors() -> list[tuple[str, Any]]:
        log("Loading deepfake detectors...")
        device = 0 if torch.cuda.is_available() else -1
        loaded: list[tuple[str, Any]] = []

        for model_id in DEEPFAKE_MODEL_CANDIDATES:
                model_id = clean_text(model_id)
                if not model_id:
                        continue
                try:
                        detector = pipeline(
                                "image-classification",
                                model=model_id,
                                device=device,
                        )
                        loaded.append((model_id, detector))
                        log(f"Loaded deepfake detector: {model_id}")
                except Exception as exc:
                        log(f"Deepfake detector unavailable ({model_id}): {exc}")

        if not loaded:
                        log("Deepfake model fallback engaged: no detector could be loaded.")
        return loaded


def load_audio_detector():
        model_id = clean_text(AUDIO_DEEPFAKE_MODEL_ID)
        if not model_id:
                return None
        try:
                detector = pipeline("audio-classification", model=model_id)
                log(f"Loaded audio detector: {model_id}")
                return detector
        except Exception as exc:
                log(f"Audio detector unavailable ({model_id}): {exc}")
                return None


def load_ai_image_detector():
        try:
                return pipeline(
                        "image-classification",
                        model="umm-maybe/AI-image-detector",
                        device=0 if torch.cuda.is_available() else -1,
                )
        except Exception as exc:
                log(f"AI image detector unavailable: {exc}")
                return None


DEEPFAKE_DETECTORS = load_deepfake_detectors()
AUDIO_DETECTOR = load_audio_detector()
AI_IMAGE_MODEL = load_ai_image_detector()


def clear_cuda_cache() -> None:
        if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_embedding(image_path: str) -> np.ndarray | None:
        img = cv2.imread(image_path)
        if img is None:
                return None

        if face_app is None:
                return None

        faces = face_app.get(img)

        if len(faces) == 0:
                try:
                        img_resized = cv2.resize(img, (640, 640))
                        faces = face_app.get(img_resized)
                except Exception:
                        return None

        if len(faces) == 0:
                return None
        return faces[0].embedding


def verify_identity(img1: str, img2: str) -> dict[str, Any]:
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        if emb1 is None or emb2 is None:
                return {"verified": False, "distance": 1.0, "similarity": 0.0, "threshold": 0.6}

        denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if denom <= 0.0:
                return {"verified": False, "distance": 1.0, "similarity": 0.0, "threshold": 0.6}

        sim = float(np.dot(emb1, emb2) / denom)
        print("Identity similarity:", sim, flush=True)

        verified = sim > 0.4
        return {
                "verified": verified,
                "distance": float(1 - sim),
                "similarity": sim,
                "threshold": 0.4,
        }


def parse_fake_score(label: str, score: float) -> float:
        lower = clean_text(label).lower()
        if any(token in lower for token in ("fake", "deepfake", "spoof", "manipulated")):
                return score
        if any(token in lower for token in ("real", "authentic", "bonafide", "genuine", "live")):
                return 1.0 - score
        return score


def parse_audio_model_score(results: Any) -> float | None:
        if not results or not isinstance(results, list):
                return None
        best = results[0]
        label = clean_text(best.get("label", "")).lower()
        score = safe_float(best.get("score"), 0.5)

        if any(token in label for token in ("fake", "spoof", "deepfake", "synthetic")):
                return max(0.0, min(score, 1.0))
        if any(token in label for token in ("real", "bonafide", "human", "authentic", "live")):
                return max(0.0, min(1.0 - score, 1.0))
        return None


def get_deepfake_score(image_path: str) -> dict[str, Any]:
        if not DEEPFAKE_DETECTORS:
                return {"score": 0.5, "label": "unknown", "source": "fallback"}

        try:
                votes: list[float] = []
                labels: list[str] = []

                for model_id, detector in DEEPFAKE_DETECTORS:
                        results = detector(image_path)
                        print(f"Deepfake model output [{model_id}]:", results, flush=True)
                        if not results:
                                continue
                        best = results[0]
                        label = str(best.get("label", "unknown"))
                        score = safe_float(best.get("score"), 0.5)
                        fake_score = parse_fake_score(label, score)
                        votes.append(max(0.0, min(fake_score, 1.0)))
                        labels.append(label)

                if not votes:
                        return {"score": 0.5, "label": "unknown", "source": "empty"}

                ensemble_score = sum(votes) / len(votes)

                clear_cuda_cache()
                return {
                        "score": round(max(0.0, min(ensemble_score, 1.0)), 4),
                        "label": labels[0] if len(labels) == 1 else f"ensemble({', '.join(labels[:2])})",
                        "source": "ensemble_model" if len(votes) > 1 else "model",
                }
        except Exception as exc:
                log(f"Deepfake model error: {exc}")
                clear_cuda_cache()
                return {"score": 0.62, "label": "fallback", "source": "error"}


def ai_generated_score(image_path: str) -> float:
        if AI_IMAGE_MODEL is None:
                return 0.0

        try:
                res = AI_IMAGE_MODEL(image_path)[0]
                label = clean_text(res.get("label", "")).lower()
                score = safe_float(res.get("score"), 0.0)

                if "ai" in label or "generated" in label or "fake" in label:
                        return max(0.0, min(score, 1.0))
                return max(0.0, min(1.0 - score, 1.0))
        except Exception:
                return 0.0


def gemini_vision_score(image_path: str) -> tuple[float, str]:
        try:
                model = get_next_gemini_model()
                if model is None:
                        return 0.0, "unavailable"

                img = Image.open(image_path)

                prompt = textwrap.dedent(
                        """
                        You are an AI forensic analyst.

                        Analyze this image for:
                        - AI-generated appearance
                        - unnatural textures
                        - symmetry artifacts
                        - lighting inconsistencies
                        - background distortions

                        Return ONLY JSON:
                        {
                          "ai_generated": true/false,
                          "confidence": 0-1,
                          "reason": "short explanation"
                        }
                        """
                ).strip()

                response = model.generate_content([prompt, img])
                text = clean_text(getattr(response, "text", ""))
                parsed = safe_json_loads(text)

                if not parsed:
                        return 0.0, "unknown"

                reason = clean_text(parsed.get("reason", ""))
                if parsed.get("ai_generated"):
                        return max(0.0, min(safe_float(parsed.get("confidence"), 0.5), 1.0)), reason
                return 0.0, reason
        except Exception as exc:
                print("Gemini vision error:", exc, flush=True)
                return 0.0, "error"


def build_local_decision(
        celebrity_name: str,
        deepfake_score: float,
        identity_match: bool,
        face_detected: bool,
        audio_risk: float,
) -> dict[str, Any]:
        risk = (
                deepfake_score * 0.5
                + (0.3 if identity_match else 0.0)
                + (0.2 if face_detected else 0.0)
        )
        risk += audio_risk * 0.1
        risk = max(0.0, min(risk, 1.0))
        misuse = bool(identity_match and deepfake_score >= DEEPFAKE_THRESHOLD)
        if not identity_match:
                misuse = False
                risk = min(risk, 0.4)
        if not face_detected:
                misuse = False
                risk = min(risk, 0.35)

        return {
                "misuse": misuse,
                "confidence": round(risk, 3),
                "reason": (
                        f"Local heuristic flagged the content because the face was{' ' if face_detected else ' not '}detected, "
                        f"identity match was {'positive' if identity_match else 'negative'}, deepfake score was {deepfake_score:.3f}, "
                        f"and audio risk was {audio_risk:.3f}."
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
        audio_risk: float,
) -> tuple[dict[str, Any], str]:
        local_json = build_local_decision(
                celebrity_name=celebrity_name,
                deepfake_score=deepfake_score,
                identity_match=identity_result["verified"],
                face_detected=face_detected,
                audio_risk=audio_risk,
        )

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
                - Audio risk: {audio_risk:.4f}

                Hard rule:
                - Set misuse=false when identity verified is false.
                - Set misuse=false when face detected is false.

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
                raw_text = call_gemini_with_retry(prompt)

                if raw_text is None:
                        raw_text = call_openai(prompt)

                if raw_text is None:
                        parsed = local_json
                        raw_text = json.dumps(local_json, indent=2)
                else:
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

        llm_text = call_gemini_with_retry(prompt)
        if llm_text is None:
                llm_text = call_openai(prompt)
        if llm_text:
                return llm_text

        return textwrap.dedent(
                f"""
                Subject: Takedown Request for Suspected Misleading Content

                Dear Platform Trust & Safety Team,

                I am requesting review and removal of content that appears to impersonate or misuse the identity of {celebrity_name}.

                Source: {source_label}
                Details: {source_details}
                AI review: {json.dumps(analysis_json, indent=2)}

                Please remove or restrict the content pending review.

                Sincerely,
                Deepfake Protection AI
                """
        ).strip()


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
                path = ensure_media_file(uploaded_image)
                if not path:
                        return None, source_label, "Upload an image to begin.", metadata
                output_path = TEST_DIR / f"scan_{run_id}.jpg"
                save_file_copy(path, output_path)
                source_details = f"Local image: {Path(path).name}"
                return str(output_path), source_label, source_details, metadata

        if source_mode == "Local Video":
                path = ensure_media_file(uploaded_video)
                if not path:
                        return None, source_label, "Upload a video to analyze frames.", metadata
                source_details = f"Local video: {Path(path).name}"
                return str(path), source_label, source_details, metadata

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


def temporal_score(frame_paths: list[str]) -> float:
        """Calculate temporal consistency score (higher = more fake/inconsistent)."""
        embeddings: list[np.ndarray] = []

        for path in frame_paths:
                emb = get_embedding(path)
                if emb is not None:
                        embeddings.append(emb)

        if len(embeddings) < 2:
                return 0.0

        diffs: list[float] = []
        for i in range(len(embeddings) - 1):
                denom = float(np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
                if denom <= 0.0:
                        diffs.append(1.0)
                else:
                        sim = float(np.dot(embeddings[i], embeddings[i + 1]) / denom)
                        diffs.append(max(0.0, min(1.0 - sim, 1.0)))

        return float(np.mean(diffs))


def identity_consistency(frame_paths: list[str], ref_path: str) -> float:
        """Check identity consistency across frames (0-1, higher = more consistent)."""
        if not frame_paths:
                return 0.0

        matches = 0
        for frame_path in frame_paths:
                try:
                        res = verify_identity(frame_path, ref_path)
                        if res["verified"]:
                                matches += 1
                except Exception:
                        pass

        return float(matches / len(frame_paths)) if frame_paths else 0.0


def build_dashboard_html(
        celebrity_name: str,
        source_label: str,
        source_details: str,
        face_detected: bool,
        face_count: int,
        identity_result: dict[str, Any],
        deepfake_result: dict[str, Any],
        audio_risk: float,
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

        explainability = analysis_json.get("explainability", {})
        temp_risk = safe_float(explainability.get("temporal_inconsistency", 0.0), 0.0)
        identity_cons = safe_float(explainability.get("identity_consistency", 1.0), 1.0)
        ai_gen = safe_float(explainability.get("ai_generated_score", explainability.get("ai_generated", 0.0)), 0.0)
        vision_gen = safe_float(explainability.get("gemini_vision", 0.0), 0.0)

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
                        card("AI Breakdown", f"DF:{deepfake_result['score']:.2f} | AU:{audio_risk:.2f}", f"Temporal:{temp_risk:.2f} | Consistency:{identity_cons:.2f}"),
                ]
        )

        dashboard_template = extract_template_block(UI_TEMPLATE, "DASHBOARD")
        if not dashboard_template.strip():
                return "<div class='error-banner'>Missing UI template: ui/templates.html</div>"

        face_crop_path = analysis_json.get("face_crop")
        if face_crop_path and os.path.exists(face_crop_path):
                face_crop_html = f"<img src='{face_crop_path}' class='face-img'/>"
        else:
                face_crop_html = "<div class='no-face'>No face detected</div>"

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
                audio_score=f"{audio_risk:.3f}",
                deepfake=f"{deepfake_result['score']:.2f}",
                ai=f"{ai_gen:.2f}",
                vision=f"{vision_gen:.2f}",
                audio=f"{audio_risk:.2f}",
                temporal=f"{temp_risk:.2f}",
                identity="1" if identity_result["verified"] else "0",
                face_crop=clean_text(analysis_json.get("face_crop", "")),
                face_crop_html=face_crop_html,
                human_explanation=clean_text(analysis_json.get("human_explanation", "")),
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

        evidence_path = suspicious_path
        audio_risk = 0.0

        if source_mode == "Local Video":
                frames = extract_multiple_frames(suspicious_path)
                if not frames:
                        message = "Could not extract frames from the uploaded video."
                        return None, f"<div class='error-banner'>{message}</div>", {}, "", message

                scores: list[float] = []
                for frame_path in frames:
                        result = get_deepfake_score(frame_path)
                        scores.append(safe_float(result.get("score"), 0.5))

                deepfake_score = sum(scores) / len(scores)
                deepfake_result = {
                        "score": round(max(0.0, min(deepfake_score, 1.0)), 4),
                        "label": f"avg_{len(scores)}_frames",
                        "source": "multi_frame_model",
                }
                evidence_path = frames[0]
                audio_risk = audio_score(suspicious_path)
                temp_risk = temporal_score(frames)
                identity_consistency_score = identity_consistency(frames, reference_path)
        else:
                deepfake_result = get_deepfake_score(suspicious_path)
                temp_risk = 0.0

        vision_score, vision_reason = gemini_vision_score(evidence_path)

        face_detected, face_count, face_box = extract_face(evidence_path)
        print("Face detected:", face_detected, "Count:", face_count, flush=True)
        ai_score = ai_generated_score(evidence_path)

        identity_source_path = evidence_path
        face_crop_path = crop_face(identity_source_path, face_box)
        boxed_path = draw_face_box(identity_source_path, face_box)
        evidence_path = boxed_path

        # Align both reference and suspicious faces before identity verification.
        ref_detected, _, ref_box = extract_face(reference_path)
        if ref_detected and ref_box is not None:
                ref_aligned = align_face(reference_path, ref_box)
        else:
                ref_aligned = reference_path

        if face_detected and face_box is not None:
                sus_aligned = align_face(identity_source_path, face_box)
        else:
                sus_aligned = identity_source_path

        identity_result = verify_identity(sus_aligned, ref_aligned)
        ref_attr = analyze_face_attributes(reference_path)
        sus_attr = analyze_face_attributes(evidence_path)
        
        if source_mode != "Local Video":
                temp_risk = 0.0
                identity_consistency_score = 1.0 if identity_result["verified"] else 0.0
        
        analysis_json, raw_analysis_text = analyze_case(
                celebrity_name=celebrity_name,
                source_label=resolved_source,
                source_details=source_details,
                deepfake_score=deepfake_result["score"],
                identity_result=identity_result,
                face_detected=face_detected,
                audio_risk=audio_risk,
        )

        analysis_json["model_used"] = "Multi-Modal Deepfake Detection Engine (Vision + Audio + Temporal AI)"
        analysis_json["face_crop"] = face_crop_path

        # Add explainability breakdown
        analysis_json["explainability"] = {
                "deepfake_signal": round(deepfake_result["score"], 4),
                "ai_generated_score": round(ai_score, 4),
                "ai_generated": round(ai_score, 4),
                "gemini_vision": round(vision_score, 4),
                "identity_match": identity_result["verified"],
                "audio_risk": round(audio_risk, 4),
                "temporal_inconsistency": round(temp_risk, 4) if source_mode == "Local Video" else 0.0,
                "identity_consistency": round(identity_consistency_score, 4),
        }

        # ---------------- FINAL DECISION ENGINE (FIXED) ----------------

        df = deepfake_result["score"]
        vision = vision_score
        sim = identity_result.get("similarity", 0.0)

        # 🔥 RULE 1: Different person → FLAG
        if sim < 0.35:
                misuse = True
                confidence = 0.65
                reason = "Face does not match reference identity"

        # 🔥 RULE 2: AI / synthetic
        elif vision > 0.5:
                misuse = True
                confidence = 0.75
                reason = "AI-generated or synthetic face detected"

        # 🔥 RULE 3: Deepfake
        elif df > 0.5:
                misuse = True
                confidence = df
                reason = "Deepfake artifacts detected"

        # 🔥 SAFE
        else:
                misuse = False
                confidence = 0.2
                reason = "No strong issues detected"

        analysis_json["misuse"] = misuse
        analysis_json["confidence"] = round(confidence, 3)
        analysis_json["reason"] = reason

        human_reason = generate_human_reason(
                identity_result,
                ref_attr,
                sus_attr,
                vision_score,
                deepfake_result["score"],
        )
        analysis_json["human_explanation"] = human_reason

        analysis_json["pipeline"] = [
                "Face Detection (InsightFace)",
                "Identity Verification",
                "Deepfake CNN Ensemble",
                "Audio Spectral Analysis",
                "Temporal Consistency Check",
                "Explainable AI Decision",
        ]

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
                audio_risk=audio_risk,
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
                f"AI-generated score: {ai_score:.4f}",
                f"Gemini vision score: {vision_score:.4f} | reason={vision_reason}",
                f"Audio risk: {audio_risk:.4f}",
                f"LLM source: {'online' if (len(GEMINI_KEYS) > 0 or OPENAI_CLIENT is not None) else 'fallback'}",
                f"Model used: {analysis_json.get('model_used', 'unknown')}",
        ]
        if metadata:
                status_line.append(f"YouTube metadata: {json.dumps(metadata, indent=2)}")
        status_line.append(f"Raw Gemini response: {raw_analysis_text}")

        return evidence_path, dashboard_html, analysis_json, notice, "\n".join(status_line)


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


def load_ui_template() -> str:
        return load_ui_text("templates.html")


def extract_template_block(template: str, block_name: str) -> str:
        start_marker = f"<!-- {block_name}_START -->"
        end_marker = f"<!-- {block_name}_END -->"
        start_index = template.find(start_marker)
        end_index = template.find(end_marker)

        if start_index == -1 or end_index == -1 or end_index <= start_index:
                raise ValueError(f"Missing template block: {block_name}")

        start_index += len(start_marker)
        return template[start_index:end_index].strip()


UI_TEMPLATE = load_ui_template()
APP_CSS = load_ui_text("styles.css")
APP_JS = load_ui_text("app.js")


def hero_html() -> str:
        return extract_template_block(UI_TEMPLATE, "HERO")


def source_help_text() -> str:
        return extract_template_block(UI_TEMPLATE, "FLOW")


if __name__ == "__main__":
        with gr.Blocks(theme="huggingface", css=APP_CSS, js=APP_JS, title=APP_NAME) as demo:
                gr.HTML("<div class='ai-cursor'></div><div class='ai-cursor-trail'></div>")
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